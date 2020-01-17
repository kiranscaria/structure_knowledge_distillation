import argparse
import logging
import os
import pdb
from torch.autograd import Variable
import os.path as osp
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import resource
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import *

from utils.criterion import CriterionDSN, CriterionOhemDSN, CriterionPixelWise, \
    CriterionAdv, CriterionAdvForG, CriterionAdditionalGP, CriterionPairWiseforWholeFeatAfterPool
from networks.pspnet_combine import Res_pspnet, BasicBlock, Bottleneck
from networks.sagan_models import Discriminator
from networks.evaluate import evaluate_main
from networks.student_resnet18 import get_psp_resnet18
from networks.teacher import get_psp_dsn_resnet101

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NetModel():
    def name(self):
        return 'kd_seg'

    def __init__(self, args):
        self.args = args
        student = get_psp_resnet18()
        # student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = args.classes_num)
        load_S_model(args, student, False)
        print_model_parm_nums(student, 'student_model')
        student.train()
        self.student = student

        teacher = get_psp_dsn_resnet101()
        # teacher = Res_pspnet(Bottleneck, [3, 4, 23, 3], num_classes = args.classes_num)
        load_T_model(teacher, args.T_ckpt_path)
        print_model_parm_nums(teacher, 'teacher_model')
        teacher.eval()
        self.teacher = teacher

        D_model = Discriminator(args.preprocess_GAN_mode, args.classes_num, args.batch_size, args.imsize_for_adv, args.adv_conv_dim)
        load_D_model(args, D_model, False)
        print_model_parm_nums(D_model, 'D_model')
        D_model.train()
        self.D_model = D_model

        self.G_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, self.student.parameters()), 'initial_lr': args.lr_g}], args.lr_g, momentum=args.momentum, weight_decay=args.weight_decay)
        self.D_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, D_model.parameters()), 'initial_lr': args.lr_d}], args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay)

        self.best_mean_IoU = args.best_mean_IoU

        self.criterion = CriterionDSN() #CriterionCrossEntropy()
        self.criterion_pixel_wise = CriterionPixelWise()
        #self.criterion_pair_wise_for_interfeat = [self.DataParallelCriterionProcess(CriterionPairWiseforWholeFeatAfterPool(scale=args.pool_scale[ind], feat_ind=-(ind+1))) for ind in range(len(args.lambda_pa))]
        self.criterion_pair_wise_for_interfeat = CriterionPairWiseforWholeFeatAfterPool(scale=args.pool_scale, feat_ind=-5)
        self.criterion_adv = CriterionAdv(args.adv_loss_type)
        if args.adv_loss_type == 'wgan-gp':
            self.criterion_AdditionalGP = CriterionAdditionalGP(D_model, args.lambda_gp)
        self.criterion_adv_for_G = CriterionAdvForG(args.adv_loss_type)
            
        self.mc_G_loss = 0.0
        self.pi_G_loss = 0.0
        self.pa_G_loss = 0.0
        self.D_loss = 0.0

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

    def set_input(self, data):
        images, labels = data
        self.images = images.to(device)
        self.labels = labels.long().to(device)
        # self.images = Variable(images)
        # self.labels = Variable(labels)

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr*((1-float(iter)/max_iter)**(power))
            
    def adjust_learning_rate(self, base_lr, optimizer, i_iter):
        args = self.args
        lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def forward(self):
        with torch.no_grad():
            self.teacher.eval()
            self.preds_T = self.teacher(self.images)
        self.student.train()
        self.preds_S = self.student(self.images)

    def student_backward(self):
        args = self.args
        G_loss = 0.0

        # pdb.set_trace()

        temp = self.criterion(self.preds_S, self.labels)
        temp_T = self.criterion(self.preds_T, self.labels)
        self.mc_G_loss = temp.item()
        G_loss = G_loss + temp
        if args.pi == True:
            temp = args.lambda_pi*self.criterion_pixel_wise(self.preds_S, self.preds_T)
            self.pi_G_loss = temp.item()
            G_loss = G_loss + temp
        if args.pa == True:
            #for ind in range(len(args.lambda_pa)):
            #    if args.lambda_pa[ind] != 0.0:
            #        temp1 = self.criterion_pair_wise_for_interfeat[ind](self.preds_S, self.preds_T, is_target_scattered = True)
            #        self.pa_G_loss[ind] = temp1.item()
            #        G_loss = G_loss + args.lambda_pa[ind]*temp1
            #    elif args.lambda_pa[ind] == 0.0:
            #        self.pa_G_loss[ind] = 0.0
            temp1 = self.criterion_pair_wise_for_interfeat(self.preds_S, self.preds_T)
            self.pa_G_loss = temp1.item()
            G_loss = G_loss + args.lambda_pa*temp1
        if self.args.ho == True:
            self.D_model.eval()
            d_out_S = self.D_model(self.preds_S)
            G_loss = G_loss + args.lambda_d*self.criterion_adv_for_G(d_out_S, d_out_S)
        G_loss.backward()
        self.G_loss = G_loss.item()

    def discriminator_backward(self):
        self.D_solver.zero_grad()
        args = self.args
        self.D_model.eval()
        d_out_T = self.D_model(self.preds_T)
        d_out_S = self.D_model(self.preds_S)
        d_loss = args.lambda_d*self.criterion_adv(d_out_S, d_out_T)

        if args.adv_loss_type == 'wgan-gp':
            d_loss += args.lambda_d*self.criterion_AdditionalGP(self.preds_S, self.preds_T)

        d_loss.backward()
        self.D_loss = d_loss.item()
        self.D_solver.step()

    def optimize_parameters(self):
        self.forward()
        self.G_solver.zero_grad()
        self.student_backward()
        self.G_solver.step()
        if self.args.ho == True:
            self.discriminator_backward()

    def evalute_model(self, model, loader, gpu_id, input_size, num_classes, whole):
        mean_IoU, IoU_array = evaluate_main(model=model, loader = loader,  
                gpu_id = gpu_id, 
                input_size = input_size, 
                num_classes = num_classes,
                whole = whole)
        return mean_IoU, IoU_array 

    def print_info(self, epoch, step):
        logging.info('step:{:5d} G_lr:{:.6f} G_loss:{:.5f}(mc:{:.5f} pixelwise:{:.5f} pairwise:{:.5f}) D_lr:{:.6f} D_loss:{:.5f}'.format(
                        step, self.G_solver.param_groups[-1]['lr'], 
                        self.G_loss, self.mc_G_loss, self.pi_G_loss, self.pa_G_loss, 
                        self.D_solver.param_groups[-1]['lr'], self.D_loss))

    def __del__(self):
        pass

    def save_ckpt(self, epoch, step, mean_IoU, IoU_array):
        torch.save(self.student.state_dict(),osp.join(self.args.snapshot_dir, 'CS_scenes_'+str(step)+'_'+str(mean_IoU)+'.pth'))  



