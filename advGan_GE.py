import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
from torch.autograd import Variable as V
from dct import *
from tqdm import tqdm
from tqdm import tqdm
from PIL import Image
import os
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from loader import ImageNet
import argparse
from torch.utils.data import DataLoader
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
)
input_csv = './dataset/images.csv'
input_dir = './dataset/images'


def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval().cuda(),)
    return model



input_csv = './dataset/images.csv'
batch_size = 10

def verify(model_name, path,adv_dir):

    model = get_model(model_name, path)

    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images)[0].argmax(1) != (gt+1)).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max,
                 save_path='./models_advGan/',
                 adv_lambda=10,
                 N=10,
                 epochs=60,
                 change_thres=[10,20,30,40,50],
                 d_ranges=[1,1,1,1,1],
                 g_ranges=[1,1,1,1,1],
                 g_lrs=[0.0001,0.0001,0.0001,0.00001,0.00001],
                 d_lrs=[0.0001,0.0001,0.0001,0.00001,0.00001],
                 rho=0.5,
                 sigma=16,
                 exp_name = 'exp0',
                 pert_lambda = 1,
                ):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)
        
        self.save_path = save_path
        self.adv_lambda=adv_lambda
        self.epochs=epochs
        self.change_thres=change_thres
        self.d_ranges=d_ranges
        self.d_ranges = [1] + self.d_ranges
        self.g_ranges=g_ranges
        self.g_ranges = [1] + self.g_ranges
        self.d_lrs=d_lrs
        self.d_lrs = [0.001] + self.d_lrs
        self.g_lrs=g_lrs
        self.g_lrs = [0.001] + self.g_lrs
        self.N = N
        self.rho = rho
        self.sigma = sigma
        self.exp_name = exp_name
        self.pert_lambda = pert_lambda

        if not os.path.exists(save_path):
            os.makedirs(save_path)
# def train_batch(self, x, labels, [5, 10, 11], [[4, 0.001], 3, 2]):
    def train_batch(self, x, labels):
        # optimize D
        for i in range(self.d_ranges[self.idx]):
            perturbation = self.netG(x)[:,:,:-1,:-1]

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -16.0/255, 16.0/255) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            loss_D_GAN = loss_D_GAN
            self.optimizer_D.step()

        # optimize G
        for i in range(self.g_ranges[self.idx]):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.reshape(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
#             logits_model = self.model(adv_images)
#             probs_model = F.softmax(logits_model, dim=1)
#             onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

#             # C&W loss function
#             real = torch.sum(onehot_labels * probs_model, dim=1)
#             other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
#             zeros = torch.zeros_like(other)
#             loss_adv = torch.max(real - other, zeros)
#             loss_adv = torch.sum(loss_adv)


            adv_images_copy = adv_images.clone()
            G = 0
            N = self.N
            rho=self.rho
            image_width=299
            sigma=self.sigma
            for n in range(N):
                gauss = torch.randn(adv_images_copy.size()[0], 3, image_width, image_width) * (sigma / 255)
                gauss = gauss.cuda()
                adv_images_copy_dct = dct_2d(adv_images_copy + gauss).cuda()
                mask = (torch.rand_like(adv_images_copy) * 2 * rho + 1 - rho).cuda()
                adv_images_copy_idct = idct_2d(adv_images_copy_dct * mask)
                adv_images_copy_idct = V(adv_images_copy_idct, requires_grad = True)

                # DI-FGSM https://arxiv.org/abs/1803.06978
                # output_v3 = model(DI(x_idct))

                output_v3 = self.model(adv_images_copy_idct)
                loss = F.cross_entropy(output_v3, labels)
                loss.backward()
                G += adv_images_copy_idct.grad.data
            G = G / N
            loss_adv = -torch.sum(adv_images * G.sign()) / adv_images.shape[0]
            # else:
            #     loss_adv = torch.sum(adv_images * G.sign()) / adv_images.shape[0]

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)
            
            # perturbation rate
            diff_transformed = adv_images - x
            
            squ_perturb_transformed = (((diff_transformed)**2).sum() / np.prod(adv_images.shape))
            abs_perturb_transformed = ((abs(diff_transformed)).sum() / np.prod(adv_images.shape))

            diff = (adv_images - x)*255
            squ_perturb = (((diff)**2).sum() / np.prod(adv_images.shape))
            abs_perturb = ((abs(diff)).sum() / np.prod(adv_images.shape))

            adv_lambda = self.adv_lambda
            pert_lambda = self.pert_lambda
            
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()
            adv_images = adv_images.detach().requires_grad_(True)
            perturbation = perturbation.detach().requires_grad_(True)



        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item(), squ_perturb.item(), abs_perturb.item(), squ_perturb_transformed.item(), abs_perturb_transformed.item()

    def train(self, train_dataloader):
        epochs = self.epochs
        self.idx = 0
        for epoch in range(1, epochs+1):

            # if epoch == 50:
            #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
            #                                         lr=0.0001)
            #     self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
            #                                         lr=0.0001)
            # if epoch == 80:
            #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
            #                                         lr=0.00001)
            #     self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
            #                                         lr=0.00001)
            
            
            if epoch > self.change_thres[self.idx]:
                if self.idx < len(self.change_thres) - 1:
                    self.idx += 1
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=self.g_lrs[self.idx])
            self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                lr=self.d_lrs[self.idx])
            
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            squ_perturb_sum = 0
            abs_perturb_sum = 0
            squ_perturb_transformed_sum = 0
            abs_perturb_transformed_sum = 0

            for i, data in enumerate(train_dataloader, start=0):
                images,_, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, squ_perturb_batch, abs_perturb_batch, squ_perturb_transformed_batch, abs_perturb_transformed_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                squ_perturb_transformed_sum += squ_perturb_transformed_batch
                abs_perturb_transformed_sum += abs_perturb_transformed_batch
                squ_perturb_sum += squ_perturb_batch
                abs_perturb_sum += abs_perturb_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.4f, loss_G_fake: %.4f,\
             \nloss_perturb: %.4f, loss_adv: %.4f, \nsqu_perturb: %.4f, abs_perturb: %.4f, \nsqu_perturb_transformed_sum: %.4f, abs_perturb_transformed_sum: %.4f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch,
                   squ_perturb_sum/num_batch, abs_perturb_sum/num_batch,
                   squ_perturb_transformed_sum/num_batch, abs_perturb_transformed_sum/num_batch))


            # save generator
            if epoch%20==0:
                if not os.path.exists(self.save_path+ 'saved_model/'):
                    os.makedirs(self.save_path+ 'saved_model/')
                netG_file_name = self.save_path + 'saved_model/netG_epoch_' + str(epoch) + '.pth'
                # if not os.path.exists(netG_file_name):
                #     os.makedirs(netG_file_name)
                torch.save(self.netG.state_dict(), netG_file_name)
                
            if epoch % 2 == 0:
                self.save(train_dataloader)
                model_names = ['tf_inception_v3','tf_inception_v4','tf_inc_res_v2','tf_resnet_v2_50','tf_resnet_v2_101','tf_resnet_v2_152','tf_ens3_adv_inc_v3','tf_ens4_adv_inc_v3','tf_ens_adv_inc_res_v2']
                models_path = './models/'
                for model_name in model_names:
                    verify(model_name, models_path,self.save_path)
            
    def generate(self, x):
        perturbation = self.netG(x)[:,:,:-1,:-1]
        adv_images = torch.clamp(perturbation, -16.0/255, 16.0/255) + x
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)
        return adv_images
    
    
    def save_image(self,images,names,output_dir):
        """save the adversarial images"""
        # if os.path.exists(output_dir)==False:
        #     os.makedirs(output_dir)

        for i,name in enumerate(names):
            img = Image.fromarray(images[i].astype('uint8'))
            img.save(output_dir+ name)
    
    def save(self,data_loader):
        for images, images_ID,  gt_cpu in tqdm(data_loader):
            gt = gt_cpu.cuda()
            images = images.cuda()
            adv_img = self.generate(images)
            adv_img_np = adv_img.detach().cpu().numpy()

            adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
            self.save_image(adv_img_np, images_ID, self.save_path)