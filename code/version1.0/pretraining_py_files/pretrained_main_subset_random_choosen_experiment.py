import argparse
from email.policy import default
import os
import random
import shutil
from termios import PARENB
import time
import warnings
import pandas as pd
import random
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
###############################################################################
parser.add_argument("--log", default='', type=str, help="saving path of the model checkpoints and the best")

parser.add_argument("-atfn","--alter-trainingfile-name", default=None, type = str)
parser.add_argument("-avfn","--alter-validationfile-name", default=None, type=str)
parser.add_argument("-s", "--subset", action='store_true')
parser.add_argument("-sp", "--subsetpath", default = False, type = str, help = 
'you must give a csv file containing' 
'the labels you want to exclude! Note that your class'
'label should be your first column in the csv file')
parser.add_argument("-r", "--randomchosenmode", action='store_true')
parser.add_argument( "--num-classes", default = 1000, type = int)
###############################################################################

best_acc1 = 0

def main():
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    #######################换了位置################################

    #以下是处理gpus 和 不同 distributed 选项之间的方案，值得深挖！
    # create model
    # update the parameters for the number of classes if necessary
    if args.subset and args.subsetpath is not None:
            args.num_classes= 1000 - len(pd.read_csv(args.subsetpath, header = None))
            
    print("when creating the model, the number of classes are")
    print(args.num_classes)

    # # Special version of alexnet for CDAN
    # if args.pretrained and args.arch == 'alexnet':
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models_CDAN.alexnet(pretrained=True, num_classes = args.num_classes)
    # elif args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True, num_classes = args.num_classes)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=False, num_classes = args.num_classes)


    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes = args.num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False, num_classes = args.num_classes)

 

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # 这里的是关键
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            # cuda（） 的实际作用
            # model = torch.nn.DataParallel(model)
            model = torch.nn.DataParallel(model).cuda()
            #model = torch.nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 换成 adam 优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr =  args.lr, weight_decay = args.weight_decay)
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size =  30, gamma = 0.1 ,verbose = True)
    

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    #######################换了位置#################################

    # Data loading code
    ###################################################
    if args.alter_trainingfile_name is not None:
        training_filename = args.alter_trainingfile_name
    else:
        training_filename = 'train'
    if args.alter_validationfile_name is not None:
        valid_filename  = args.alter_validationfile_name
    else:
        valid_filename = 'val'
    ###################################################

    traindir = os.path.join(args.data, training_filename)
    valdir = os.path.join(args.data, valid_filename)
    ####################################################
    print("--"*20)
    print("training directory:" + traindir + "\n")
    print("validating directory:" + valdir + "\n")
    print("--"*20)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset_initial = datasets.ImageFolder(traindir, 
    transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    print(train_dataset_initial)
    print("--"*20)
    
    ######################################################################################################################
    # extend the functionality to include the subset model training_filename
    valid_dataset_initial = datasets.ImageFolder(valdir, 
    transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    print(valid_dataset_initial)
    print("--"*20)

    if args.subset:
        if args.subsetpath is not None:
            masked_label_list= pd.read_csv(args.subsetpath, header = None)
            print(masked_label_list)
            if args.randomchosenmode:
                # fix the random seed to produce the replicable sampling results
                random.seed(1000)

                # select the classes after excluding the masked classes 
                exclude_masked_classes = [one_class for one_class in train_dataset_initial.classes if one_class not in list(masked_label_list[0])]

                # assign the number of classes to args 
                args.num_classes = len(exclude_masked_classes)
                print("+---"*20)
                print(len(exclude_masked_classes))
                print(len([train_dataset_initial.class_to_idx[c] for c in list(masked_label_list[0])]))
                print("+---"*20)

                random_selected_classes = random.sample(exclude_masked_classes, len(masked_label_list))
                chosen_classes_labels_indices = [train_dataset_initial.class_to_idx[each] for each in train_dataset_initial.classes if each not in random_selected_classes]
                print(len(chosen_classes_labels_indices))

                classes_dict = {x:i for i,x in enumerate(chosen_classes_labels_indices)}

                chosen_classes_labels_names = [each for each in train_dataset_initial.classes if each not in random_selected_classes]
                print("chosen_classes_labels_names")
                print(chosen_classes_labels_names)

                # Save the random selected_classes to a csv file
                random_selected_classes_labels_names_df = pd.DataFrame(random_selected_classes)
                print(random_selected_classes_labels_names_df)
                random_selected_classes_labels_names_df.to_csv(args.log + '/random_selected_classes_labels.csv', index = False, header = False)

                # Find all relevant indices in the training and validating sets
                chosen_index_train = [index for index in range(len(train_dataset_initial)) if train_dataset_initial.imgs[index][1] in chosen_classes_labels_indices]
                chosen_index_valid = [index for index in range(len(valid_dataset_initial)) if valid_dataset_initial.imgs[index][1] in chosen_classes_labels_indices]
            else:
                # assigned the number of classes to args 
                masked_classes = [train_dataset_initial.class_to_idx[c] for c in list(masked_label_list[0])]
                args.num_classes = len(train_dataset_initial.classes) - len(masked_classes)
                print("+---"*20)
                print(args.num_classes)
                print(len(masked_classes))
                print("+---"*20)

                chosen_classes_labels_names = [each for each in train_dataset_initial.classes if each not in list(masked_label_list[0])]
                print("chosen_classes_labels_names")
                print(chosen_classes_labels_names)
                
                chosen_index_train = [index for index in range(len(train_dataset_initial)) if train_dataset_initial.imgs[index][1] not in masked_classes]
                chosen_index_valid = [index for index in range(len(valid_dataset_initial)) if valid_dataset_initial.imgs[index][1] not in masked_classes]

                chosen_classes_labels_indices = [train_dataset_initial.class_to_idx[each] for each in train_dataset_initial.classes if each not in list(masked_label_list[0])]
                classes_dict = {x:i for i,x in enumerate(chosen_classes_labels_indices)}

            train_dataset = torch.utils.data.Subset(train_dataset_initial, chosen_index_train)
            valid_dataset = torch.utils.data.Subset(valid_dataset_initial, chosen_index_valid)
            print(len(chosen_index_train))
            print("train_datast length is %d" % (len(train_dataset)))
            print(len(chosen_index_valid))
            print("valid_dataset length is %d" % (len(valid_dataset)))
        else:
            warnings.warn('Since you do not specify the csv file for the class labels you are going to mask, so no subset model training will be used in this case!')
    else:
        train_dataset, valid_dataset = train_dataset_initial,valid_dataset_initial
        classes_dict = {i:i for i in range(len(train_dataset.classes))}

    print(classes_dict)

    ##############################################################################################
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    print("training set")
    print(len(train_loader))

    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print("validating  set")
    print(len(val_loader))



    #######################换了位置################################

    # #以下是处理gpus 和 不同 distributed 选项之间的方案，值得深挖！
    # # create model
    # # update the parameters for the number of classes if necessary
    # if args.subset and args.subsetpath is not None:
    #         args.num_classes= 1000 - len(pd.read_csv(args.subsetpath, header = None))
            
    # print("when creating the model, the number of classes are")
    # print(args.num_classes)
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True, num_classes = args.num_classes)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=False, num_classes = args.num_classes)

 

    # if not torch.cuda.is_available():
    #     print('using CPU, this will be slow')
    # elif args.distributed:
    #     # For multiprocessing distributed, DistributedDataParallel constructor
    #     # should always set the single device scope, otherwise,
    #     # DistributedDataParallel will use all available devices.
    #     if args.gpu is not None:
    #         torch.cuda.set_device(args.gpu)
    #         model.cuda(args.gpu)
    #         # When using a single GPU per process and per
    #         # DistributedDataParallel, we need to divide the batch size
    #         # ourselves based on the total number of GPUs we have
    #         args.batch_size = int(args.batch_size / ngpus_per_node)
    #         args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     else:
    #         model.cuda()
    #         # DistributedDataParallel will divide and allocate batch_size to all
    #         # available GPUs if device_ids are not set
    #         model = torch.nn.parallel.DistributedDataParallel(model)
    # elif args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)
    # else:
    #     # 这里的是关键
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         # cuda（） 的实际作用
    #         # model = torch.nn.DataParallel(model)
    #         model = torch.nn.DataParallel(model).cuda()
    #         #model = torch.nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))


    # # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         if args.gpu is None:
    #             checkpoint = torch.load(args.resume)
    #         else:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.gpu)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True


    #######################换了位置#################################



    if args.evaluate:
        validate(val_loader, model, criterion, args, classes_dict = classes_dict)
        return


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # 换掉 原装的lr sched
        # adjust_learning_rate(optimizer, epoch, args)
       
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, epoch, classes_dict = classes_dict)


        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch, classes_dict = classes_dict)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        #  Adjusting the saving location of checkpoints and best models
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'num_classes':args.num_classes,
            }, is_best, saving_dir = args.log)
        my_lr_scheduler.step()



def train(train_loader, model, criterion, optimizer, epoch, args, epoch_num,  classes_dict = None ):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    if args.log == '':
        writer = SummaryWriter(comment = 'training')
    else:
        writer = SummaryWriter(log_dir = args.log + '/logtrainresults/', comment = 'training')
     

    # switch to train mode
    model.train()
    total_losses = 0.0
    total_correct_count_top1= 0
    total_correct_count_top5= 0

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = torch.tensor([classes_dict[x.item()] for x in target])
        if torch.cuda.is_available():
            ###################################
            # target_map_before = torch.tensor([classes_dict[x.item()] for x in target])
            ###################################
            target = target.cuda(args.gpu, non_blocking=True)
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        total_losses += loss
        total_correct_count_top1 += acc1
        total_correct_count_top5 += acc5
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # compute gradient and do SGD step
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


    writer.add_scalar('Loss/train',total_losses/len(train_loader), epoch_num)
    writer.add_scalar('Accuracy/top1/train', total_correct_count_top1/len(train_loader), epoch_num)
    writer.add_scalar('Accuracy/top5/train',total_correct_count_top5/len(train_loader) , epoch_num)
    writer.flush()
    writer.close()

def validate(val_loader, model, criterion,args, epoch_num = 0, classes_dict = None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')



    if args.log == '':
        writer = SummaryWriter(comment = 'validating')
    else:
        writer = SummaryWriter(log_dir = args.log + "/logtestresults/", comment = 'validating')
    # switch to evaluate mode
    model.eval()
    total_losses = 0.0
    total_correct_count_top1= 0
    total_correct_count_top5= 0
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = torch.tensor([classes_dict[x.item()] for x in target])
            if torch.cuda.is_available():
            ###################################
                # target_map_before = torch.tensor([classes_dict[x.item()] for x in target])
            ###################################
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_losses += loss
            total_correct_count_top1 += acc1
            total_correct_count_top5 += acc5
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        progress.display_summary()
        writer.add_scalar('Loss/test',total_losses/len(val_loader), epoch_num)
        writer.add_scalar('Accuracy/top1/test', total_correct_count_top1/len(val_loader), epoch_num)
        writer.add_scalar('Accuracy/top5/test',total_correct_count_top5/len(val_loader) , epoch_num)
        writer.flush()
        writer.close()


    return top1.avg


##################################################################
#  Adjusting the saving location of checkpoints and best models

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', saving_dir = ''):
    torch.save(state, saving_dir + filename)
    if is_best:
        shutil.copyfile(saving_dir + filename, saving_dir + 'model_best.pth.tar')

##################################################################

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
