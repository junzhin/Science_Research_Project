## we conduct the DA methods following https://github.com/thuml/Transfer-Learning-Library, which gives better results than the common settings.

import torch
import os
import math
import time
from .base_solver import BaseSolver
import torch.nn.functional as F
from utils.utils import AverageMeter, to_cuda, accuracy, accuracy_for_each_class, get_random_recover_index
from dalib.ssl.temporal_ensemble import consistency_loss, ImageClassifierHead


class Solver(BaseSolver):
    def __init__(self, classifier, dataloaders, args, **kwargs):
        super(Solver, self).__init__(classifier, dataloaders, args, **kwargs)

        self.total_iters = self.args.epochs * self.args.iters_per_epoch
        self.gamma = 0.001
        self.decay_rate = 0.75
        self.build_optimizer()
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
        assert not args.strongaug  ## StrongAug is NOT expected.

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
            (source_data, source_data2), source_gt = self.get_samples('source')
            (target_data_weak, target_data_weak_2), _ = self.get_samples('target')

            ## In Pi model, a warm up weight of exp(-5(1-self.iters / self.total_iters) ** 2) is adopted in the original paper, which changes 0->1
            ## However, we adopt the following strategy, which also changes 0->1, following DANN
            coeff = 2 / (1 + math.exp(-1 * 10 * self.iters / self.total_iters)) - 1
            data_time.update(time.time() - end)

            source_data = source_data.cuda(self.args.gpu, non_blocking=True)
            source_data2 = source_data2.cuda(self.args.gpu, non_blocking=True)
            target_data_weak = target_data_weak.cuda(self.args.gpu, non_blocking=True)
            target_data_weak_2 = target_data_weak_2.cuda(self.args.gpu, non_blocking=True)
            source_gt = source_gt.cuda(self.args.gpu, non_blocking=True)


            if self.args.regular_only_feature:  ###
                source_num = source_data.size(0)
                target_num = target_data_weak.size(0)

                data_concate = torch.cat((source_data, source_data2, target_data_weak, target_data_weak_2), dim=0)
                logit_concate, _ = self.classifier(data_concate)
                prob_concate = F.softmax(logit_concate, dim=1)
                prob_s_all = prob_concate[: source_num * 2, :]
                prob_t_all = prob_concate[source_num * 2:, :]
                prob_s = prob_s_all[:source_num, :]
                prob_s2 = prob_s_all[source_num:, :]
                prob_t = prob_t_all[:target_num, :]
                prob_t2 = prob_t_all[target_num:, :]

                ssl_loss = consistency_loss(prob_s, prob_s2) + consistency_loss(prob_t, prob_t2)
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
                loss = cls_loss + ssl_loss * coeff * self.args.trade_off

            else:
                source_num = source_data.size(0)
                target_num = target_data_weak.size(0)
                data_concate = torch.cat((source_data, source_data2, target_data_weak, target_data_weak_2), dim=0)
                logit_concate, _ = self.classifier(data_concate)
                prob_concate = F.softmax(logit_concate, dim=1)
                prob_s_all = prob_concate[: source_num * 2, :]
                prob_t_all = prob_concate[source_num * 2:, :]
                prob_s = prob_s_all[:source_num, :]
                prob_s2 = prob_s_all[source_num:, :]
                prob_t = prob_t_all[:target_num, :]
                prob_t2 = prob_t_all[target_num:, :]
                ssl_loss = consistency_loss(prob_s, prob_s2) + consistency_loss(prob_t, prob_t2)
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



