from loader import ImageNet
import torchvision
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from Normalize import Normalize
import argparse
from tqdm import tqdm
from PIL import Image
import pretrainedmodels
device = "cuda" if torch.cuda.is_available() else "cpu"
transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)
device = "cuda" if torch.cuda.is_available() else "cpu"
def set_seed(seed):
    # Set random seed for torch CPU
    torch.manual_seed(seed)
    # Set random seed for all GPUs
    torch.cuda.manual_seed_all(seed)
    # Set random seed for numpy
    np.random.seed(seed)
    # Set deterministic mode for CuDNN to True
    torch.backends.cudnn.deterministic = True
    # Disable CuDNN benchmarking to ensure deterministic results
    torch.backends.cudnn.benchmark = False
    # Set random seed for Python hash
    random.seed(seed)
    # Set the PYTHONHASHSEED environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(2023)

parser = argparse.ArgumentParser()
parser.add_argument('--adv_lambda', type=float, default=10.0)
parser.add_argument('--pert_lambda', type=float, default=1.0)
parser.add_argument('--model', type=str, default="inceptionv3") #inceptionv3 inceptionv4 inceptionresnetv2 resnet152
parser.add_argument('--N',type=int,default=10)
parser.add_argument('--rho', type=float, default=0.5)
parser.add_argument('--epochs',type=int,default=60)
parser.add_argument('--sigma',type=int,default=16)
parser.add_argument('--change_thres',nargs='+',default=[0,50,80])
parser.add_argument('--d_ranges',nargs='+',default=[3,3,3])
parser.add_argument('--g_ranges',nargs='+',default=[1,1,1])
parser.add_argument('--d_lrs',nargs='+',default=[0.001,0.0001,0.00001])
parser.add_argument('--g_lrs',nargs='+',default=[0.001,0.0001,0.00001])
parser.add_argument('--save_path',type=str,default="output/exp1/")
parser.add_argument('--exp_name',type=str,default="baseline")
args = parser.parse_args()
args.change_thres = [int(t) for t in args.change_thres]
args.d_ranges = [int(t) for t in args.d_ranges]
args.g_ranges = [int(t) for t in args.g_ranges]
args.d_lrs = [float(t) for t in args.d_lrs]
args.g_lrs = [float(t) for t in args.g_lrs]
print(args)

if 'baseline' in args.exp_name:
    from advGan_baseline import AdvGAN_Attack
elif 'GE' in args.exp_name:
    from advGan_GE import AdvGAN_Attack

model = eval(f"pretrainedmodels.{args.model}")

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

model = torch.nn.Sequential(Normalize(mean, std),model(num_classes=1000, pretrained='imagenet').eval().to(device))

X = ImageNet("./dataset/images", "./dataset/images.csv", transforms)

data_loader = DataLoader(X, batch_size=10, shuffle=False, pin_memory=True, num_workers=8)


attack = AdvGAN_Attack(device,model,1000,3,0,1,save_path=args.save_path,
                 adv_lambda=args.adv_lambda,N=args.N,epochs=args.epochs,change_thres=args.change_thres,
                       d_ranges=args.d_ranges,g_ranges=args.g_ranges,d_lrs=args.d_lrs,g_lrs=args.g_lrs,exp_name=args.exp_name,rho=args.rho,pert_lambda=args.pert_lambda)

attack.train(data_loader)