## we conduct the DA methods following https://github.com/thuml/Transfer-Learning-Library, which gives better results than the common settings.
import torch
import os
import math
import time
from .base_solver import BaseSolver
import torch.nn.functional as F
from dalib.ssl.temporal_ensemble import consistency_loss, ImageClassifierHead
from utils.utils import AverageMeter, to_cuda, accuracy, accuracy_for_each_class, get_random_recover_index
import copy



class Solver(BaseSolver):
    def __init__(self, classifier, dataloaders, args, **kwargs):
        super(Solver, self).__init__(classifier, dataloaders, args, **kwargs)
        self.classifier_ema = copy.deepcopy(self.classifier)
        self.total_iters = self.args.epochs * self.args.iters_per_epoch
        self.gamma = 0.001
        self.decay_rate = 0.75
        self.build_optimizer()
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])

    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.classifier.parameters(), self.classifier_ema.parameters()):
            param_eval.copy_(param_eval * self.args.ema_m + param_train.detach() * (1 - self.args.ema_m))

        for buffer_train, buffer_eval in zip(self.classifier.buffers(), self.classifier_ema.buffers()):
            buffer_eval.copy_(buffer_train)


    def update_network(self, **kwargs):
        self.classifier.train()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses_all = AverageMeter()
        losses_cls = AverageMeter()
        losses_ssl = AverageMeter()
        prec1 = AverageMeter()
        end = time.time()

        for i in range(self.iters_per_epoch):
            if self.args.lr_schedule == 'dalib':
                self.lr_sheduler_f.step()
                self.lr_sheduler_g.step()
            elif self.args.lr_schedule == 'inv':
                self.lr_sheduler_f.step(self.iters)
                self.lr_sheduler_g.step(self.iters)
            (source_data, _), source_gt = self.get_samples('source')
            (target_data_weak, _), _ = self.get_samples('target')

            ## In Pi model, a warm up weight of exp(-5(1-self.iters / self.total_iters) ** 2) is adopted in the original paper, which changes 0->1
            ## However, we adopt the following strategy, which also changes 0->1, since we initialize the model with weight pre-training on the ImageNet dataset.
            coeff = 2 / (1 + math.exp(-1 * 10 * self.iters / self.total_iters)) - 1
            data_time.update(time.time() - end)
            source_data = source_data.cuda(self.args.gpu, non_blocking=True)
            target_data_weak = target_data_weak.cuda(self.args.gpu, non_blocking=True)
            source_gt = source_gt.cuda(self.args.gpu, non_blocking=True)

            if self.args.regular_only_feature:  ###
                data_concate = torch.cat((source_data, target_data_weak), dim=0)
                logits_concate, _ = self.classifier(data_concate)
                prob_concate = F.softmax(logits_concate, dim=1)
                with torch.no_grad():
                    logits_concate_teacher, _ = self.classifier_ema(data_concate)
                    prob_concate_teacher = F.softmax(logits_concate_teacher, dim=1)
                ssl_loss = consistency_loss(prob_concate, prob_concate_teacher)
                loss_g = ssl_loss * coeff * self.args.trade_off
                self.optimizer_f.zero_grad()
                self.optimizer_g.zero_grad()
                loss_g.backward()  ## acculate the gradient
                self.optimizer_f.zero_grad()
                # self.optimizer_g.zero_grad()  ## ssl loss only on feature
                logits_s, _ = self.classifier(source_data)
                cls_loss = F.cross_entropy(logits_s, source_gt)
                loss_f = cls_loss
                loss_f.backward()
                self.optimizer_f.step()
                self.optimizer_g.step()
                loss = cls_loss + ssl_loss * self.args.trade_off
            else:
                data_concate = torch.cat((source_data, target_data_weak), dim=0)
                logits_concate, _ = self.classifier(data_concate)
                prob_concate = F.softmax(logits_concate, dim=1)
                with torch.no_grad():
                    logits_concate_teacher, _ = self.classifier_ema(data_concate)
                    prob_concate_teacher = F.softmax(logits_concate_teacher, dim=1)
                ssl_loss = consistency_loss(prob_concate, prob_concate_teacher)

                loss_g = ssl_loss * coeff * self.args.trade_off
                self.optimizer_f.zero_grad()
                self.optimizer_g.zero_grad()
                loss_g.backward()  ## acculate the gradient
                logits_s, _ = self.classifier(source_data)
                cls_loss = F.cross_entropy(logits_s, source_gt)
                loss_f = cls_loss
                loss_f.backward()
                self.optimizer_f.step()
                self.optimizer_g.step()
                loss = cls_loss + ssl_loss * self.args.trade_off

            prec1_iter = accuracy(logits_s, source_gt)
            prec1.update(prec1_iter, source_data.size(0))

            self._eval_model_update()
            losses_cls.update(cls_loss.item(), source_data.size(0))
            losses_ssl.update(ssl_loss.item(), target_data_weak.size(0))
            losses_all.update(loss.item(), source_data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            self.iters += 1
            if i % 10 == 0:
                print("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LCls:%3f, Lssl:%3f, LossAll:%3f, Acc1:%3f" % \
                      (self.iters, self.epoch, self.args.epochs, data_time.avg, batch_time.avg, losses_cls.avg, losses_ssl.avg, losses_all.avg,  prec1.avg))


        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LCls:%3f, Lssl:%3f, LossAll:%3f, Acc1:%3f" % \
                      (self.iters, self.epoch, self.args.epochs, data_time.avg, batch_time.avg, losses_cls.avg, losses_ssl.avg, losses_all.avg,  prec1.avg))
        log.close()
