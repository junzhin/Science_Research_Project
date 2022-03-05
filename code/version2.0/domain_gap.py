import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

from sklearn.model_selection import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from collections import OrderedDict as OD
from torchvision.datasets import *
sys.path.append('../../..')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args:argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        cudnn.benchmark = False

    
    # Data loading code
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    Resize = ResizeImage(256)
    Crop = T.RandomResizedCrop(224)

    train_transform = T.Compose([   
    Resize,
    T.CenterCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalize
    ])

    val_transform = T.Compose([
        Resize,
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # loading the data and cv into n fold cross validation
    dataset = ImageFolder(args.root, transform = train_transform)
    kf = KFold(n_splits=args.cv)
    spliters = kf.split(dataset)   

    # create model
    if args.local:
        print("Loading the local alexnet pretrained model weights!")
        state = torch.load(args.local_pretrained_path)
        state_dict = state['state_dict']

        if "num_classes" in state.keys():
            num_classes = state["num_classes"]
        else:
            # default value of number of classes
            num_classes = 1000

        backbone = models.__dict__[args.arch](pretrained=True, num_classes = num_classes)

        print(state_dict.keys())
        # 处理不同 pretrained model weights 的前缀,使其兼容
        state = OD([(key.split("module.")[-1], state_dict[key]) for key in state_dict])
        print(state.keys())

        backbone.load_state_dict(state, strict = False)
    else:
        print("=> using pre-trained model from pytorch website '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)

    
    classifier = ImageClassifier(backbone, num_classes=2,bottleneck_dim=256).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(),args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))


    # start training
    best_acc1 = 0.
    for fold,(train_idx,val_idx) in enumerate(spliters):
        print('------------fold no---------{}----------------------'.format(fold))

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle= True,
        num_workers=2, pin_memory=True)


        val_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)
        test_loader = val_loader

        train_source_iter = ForeverDataIterator(train_loader)


        for epoch in range(args.epochs):
            # train for one epoch
            train(train_source_iter, classifier, optimizer,lr_scheduler, epoch, args)

            
            if epoch % 1 == 0:
                # evaluate on validation set
                acc1 = validate(val_loader, classifier, args, epoch=epoch)

                # remember best acc@1 and save checkpoint
                torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
                if acc1 > best_acc1:
                    shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
                best_acc1 = max(acc1, best_acc1)
                print("DomainACC = {:3.1f}, best_DomainACC = {:3.1f}".format(acc1, best_acc1))

        print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(val_loader, classifier, args, epoch = 0)
    print("val_DomainACC = {:3.1f}".format(acc1))
    acc1 = validate(test_loader, classifier, args, epoch = 0)
    print("test_DomainACC = {:3.1f}".format(acc1))

    logger.close()    



def train(data_loader: ForeverDataIterator, model: ImageClassifier, 
        optimizer: SGD,lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
        
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    total_losses = 0.0
    total_domain_accs= 0.0
    
    writer = SummaryWriter(log_dir = args.log + '/logtrainresults/', comment = 'training')
    
    end = time.time()

    for i in range(args.iters_per_epoch):
        images, labels_s = next(data_loader)

        images = images.to(device)

        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, f = model(images)


        cls_loss = F.cross_entropy(y, labels_s)
        loss = cls_loss
        domain_acc = accuracy(y, labels_s)[0]

        # update the log results:
        total_losses += loss.item()
        total_domain_accs += domain_acc.item()

        losses.update(loss.item(), images.size(0))
        domain_accs.update(domain_acc.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
    writer.add_scalar('Loss/train',total_losses/args.iters_per_epoch, epoch)
    writer.add_scalar('Accuracy/domain_accs/train', total_domain_accs/args.iters_per_epoch, epoch)
    writer.flush()
    writer.close()

    return


def validate(val_loader: DataLoader, model: ImageClassifier, 
args: argparse.Namespace, epoch: int) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses,domain_accs],
        prefix='Test: ')


    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    total_losses = 0.0
    total_domain_accs = 0.0

    writer = SummaryWriter(log_dir = args.log + '/logtestresults/', comment = 'validating')
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, f = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            domain_acc = accuracy(output, target)[0]

            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            domain_accs.update(domain_acc.item(), images.size(0))


            # update the log results
            total_losses += loss.item()
            total_domain_accs += domain_acc.item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * DomainACC@1 {domain_accs:.3f}'
              .format(domain_accs=domain_accs))
        if confmat:
            print(confmat.format(classes))

        writer.add_scalar('Loss/test',total_losses/len(val_loader), epoch)
        writer.add_scalar('Accuracy/domain_accs/train', total_domain_accs/args.iters_per_epoch, epoch)
        writer.flush()
        writer.close()

    return domain_accs.avg



if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    # parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
    #                     help='dataset: ' + ' | '.join(dataset_names) +
    #                          ' (default: Office31)')
    # parser.add_argument('-s', '--source', help='source domain(s)')
    # parser.add_argument('-t', '--target', help='target domain(s)')
    # parser.add_argument('--center-crop', default=False, action='store_true',
    #                     help='whether use center crop during training')
    # parser.add_argument('--data_processing', type=str, default="ours", choices=['ours', 'dalib'], help='type of data transforms')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    # parser.add_argument('--bottleneck-dim', default=256, type=int,
    #                     help='Dimension of bottleneck')
    # parser.add_argument('--trade-off', default=1., type=float,
    #                     help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true', default = True,
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    # parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
    #                     help="When phase is 'test', only test the model."
    #                          "When phase is 'analysis', only analysis the model.")


    ######################################
    parser.add_argument('--pretrained', default=True, help='specify the boolean value ofr pretained model')
    parser.add_argument('--local', action='store_true', default=False, help='to inicate if the local pretrained model exists')
    parser.add_argument('--local-pretrained-path', default='',type = str, help='the path to local pre-trained-model')
    parser.add_argument('--checkmodel_logsave', default=False, action = 'store_true', help = "indicate if the training will save the best model and the checkpoint weights in case of limited storage space.")
    ######################################
    args = parser.parse_args()

    main(args)


