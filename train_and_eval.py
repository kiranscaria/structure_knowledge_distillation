import sys
from utils.train_options import TrainOptions
from networks.kd_model import NetModel
import logging
import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
#from dataset.datasets import CSDataSet
# from dataset.datasets import PFCNDataSet
from dataset.dataset_small import SmallCSDataset as CSDataSet
import numpy as np
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
args = TrainOptions().initialize()
h, w = map(int, args.input_size.split(','))
trainloader = data.DataLoader(CSDataSet(path="dataset/data/small_cityscapes/cityscapes/train"), 
    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
valloader = data.DataLoader(CSDataSet(path="dataset/data/small_cityscapes/cityscapes/train"), 
    batch_size=1, shuffle=False, pin_memory=True)
# trainloader = data.DataLoader(PFCNDataSet(args.data_dir, './dataset/list/pfcn/train.txt', max_iters=args.num_steps*args.batch_size, crop_size=(h, w), scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN ), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
# valloader = data.DataLoader(PFCNDataSet(args.data_dir, './dataset/list/pfcn/test.txt', crop_size=(800, 600), scale=False, mirror=False, mean=IMG_MEAN), batch_size=1, shuffle=False, pin_memory=True)
save_steps = int(2975/args.batch_size)
model = NetModel(args)
for epoch in range(args.start_epoch, args.epoch_nums):
    for step, data in enumerate(trainloader, args.last_step+1):
        #model.adjust_learning_rate(args.lr_g, model.G_solver, step)
        #model.adjust_learning_rate(args.lr_d, model.D_solver, step)
        model.set_input(data)
        model.optimize_parameters()
        model.print_info(epoch, step)
        if (step > 1) and ((step % save_steps == 0) and (step > args.num_steps - 1000)) or (step == args.num_steps - 1):
            mean_IU, IU_array = model.evalute_model(model.student, valloader, '0', '512,512', 2, True)
            model.save_ckpt(epoch, step, mean_IU, IU_array)
            logging.info('[val 128, 128] mean_IU:{:.6f}  IU_array:{}'.format(mean_IU, IU_array))


