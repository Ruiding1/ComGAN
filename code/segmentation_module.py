##############
# This is first going
# work hard!!
# best wish
#################

from scipy import linalg
from scipy.optimize import linear_sum_assignment
from config import cfg
import os
import time
from PIL import Image
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn

import torch.optim as optim
import os
import torchvision.utils as vutils
from model import G_NET_of_DSComGAN, MASK_D, FORE_D, BACK_D, IMAGE_D
from datasets import get_dataloader
import datasets
from datasets import *
import random
from utils import *
from itertools import chain
from copy import deepcopy
from tensorboardX import summary
from tensorboardX import FileWriter
import torchvision.transforms as transforms
cudnn.benchmark = True
from net.Unet import U_Net
import yaml

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda:" + cfg.GPU_ID)
gpus = [int(ix) for ix in cfg.GPU_ID.split(',')]



############################################################

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def define_optimizers(netG, netsD, Unet):
    # define optimizer for G and D

    optimizersD = []
    for i in range(len(netsD)):
        if i == 3 or i== 0:
            optimizersD.append(optim.Adam(netsD[i].parameters(), lr=2e-4, betas=(0.5, 0.999)))
        else:
            optimizersD.append(None)

    params = chain(netG.parameters(), netsD[0].parameters(), netsD[1].parameters(), netsD[2].parameters())
    optimizerGE = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))

    paprams = chain(Unet.parameters())
    optimizerU = optim.Adam(paprams, lr=2e-4, betas=(0.5, 0.999))

    return optimizersD, optimizerGE, optimizerU

def evaluate(netEncM, loader, device):
    nbIter = 0
    iou_s = 0
    dice_s = 0
    for xLoad, mLoad in loader:
        xData = xLoad.to(device)
        mData = mLoad.to(device)
        mPred = netEncM(xData)
        nbIter += 1

        bs =xData.size()[0]
        pred_f = mPred >= 0.5
        pred_b = mPred < 0.5
        gt = mData

        iou = torch.max(
            (pred_f * gt).view(bs, -1).sum(dim=-1) / \
            ((pred_f + gt) >= 1).view(bs, -1).sum(dim=-1),
            (pred_b * gt).view(bs, -1).sum(dim=-1) / \
            ((pred_b + gt) >= 1).view(bs, -1).sum(dim=-1)

        )

        dice = torch.max(
            2 * (pred_f * gt).view(bs, -1).sum(dim=-1) / \
            (pred_f.view(bs, -1).sum(dim=-1) + gt.view(bs, -1).sum(dim=-1)),
            2 * (pred_b * gt).view(bs, -1).sum(dim=-1) / \
            (pred_b.view(bs, -1).sum(dim=-1) + gt.view(bs, -1).sum(dim=-1))
        )
        iou_s += iou.mean().item()
        dice_s += dice.mean().item()

    return iou_s / nbIter, dice_s / nbIter


def load_network():
    # prepare G net

    netG = G_NET_of_DSComGAN().to(device)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    path = cfg.TEST.NET_G
    state_dict = torch.load(path)
    netG.load_state_dict(state_dict)
    toggle_grad(netG, False)
    netG = netG.eval()

    Unet = model_unet(U_Net, 3, 1).to(device)
    Unet = torch.nn.DataParallel(Unet, device_ids=gpus)


    netsD = [ MASK_D(), FORE_D(), BACK_D() , IMAGE_D()]
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)

    return netG, netsD, Unet

def save_model(netG, D0, D1, D2, D3, Unet, epoch, model_dir):
    torch.save(netG.state_dict(), '%s/G_%d.pth' % (model_dir, epoch))
    torch.save(D0.state_dict(), '%s/D0_%d.pth' % (model_dir, epoch))
    torch.save(D1.state_dict(), '%s/D1_%d.pth' % (model_dir, epoch))
    torch.save(D2.state_dict(), '%s/D2_%d.pth' % (model_dir, epoch))
    torch.save(D3.state_dict(), '%s/D3_%d.pth' % (model_dir, epoch))
    torch.save(Unet.state_dict(), '%s/Unet_%d.pth' % (model_dir, epoch))



class BinaryLoss(nn.Module):

    def __init__(self, loss_weight):
        super(BinaryLoss, self).__init__()
        self.loss_weight = loss_weight

    @staticmethod
    def binary_entropy(p):
        return -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)

    def __call__(self, mask):
        return self.loss_weight * self.binary_entropy(mask).mean()


class CrossEntropy():
    def __init__(self):
        self.code_loss = nn.CrossEntropyLoss()

    def __call__(self, prediction, label):
        if label.max(dim=1)[0].min() == 1:
            return self.code_loss(prediction, torch.nonzero(label.long())[:, 1])
        else:
            log_prediction = torch.log_softmax(prediction, dim=1)
            return (- log_prediction * label).sum(dim=1).mean(dim=0)


class Trainer(object):
    def __init__(self, output_dir):
        # make dir for all kinds of output

        self.model_dir = os.path.join(output_dir, 'Model')
        os.makedirs(self.model_dir)
        self.image_dir = os.path.join(output_dir, 'Image')
        os.makedirs(self.image_dir)
        self.opt_dir = os.path.join(output_dir, 'Opt')
        os.makedirs(self.opt_dir)

        # make dataloader and code buffer
        self.dataloader = get_dataloader()

        # other variables
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.patch_stride = 4.0
        self.n_out = 24
        self.recp_field = 34

        self.real_labels = torch.ones_like(torch.randn(self.batch_size, 1, 1, 1)).to(device)
        self.fake_labels = torch.zeros_like(torch.randn(self.batch_size, 1, 1, 1)).to(device)


    def prepare_data(self, data):

        _, real_img, real_c, _, _ = data

        real_img = real_img.to(device)

        real_z = torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM).normal_(0, 1).to(device) * 4
        real_c = real_c.to(device)

        real_c_inv = real_c

        return real_img, real_z, real_c, real_c_inv

    def train_Dnet(self, idx):  # ADV loss

        # choose net and opt
        netD, optD = self.netsD[idx], self.optimizersD[idx]
        netD.zero_grad()
        fake_img = self.mask_real.repeat(1,3,1,1).detach()
        real_img = self.fake_imgs_list[1].repeat(1,3,1,1).detach()
        fake_img_1 = self.mask_fake.repeat(1,3,1,1).detach()

        _, self.real_prediction = netD(real_img)
        _, self.fake_prediction = netD(fake_img_1)
        real_prediction_loss = self.RF_loss(self.real_prediction, self.real_labels)
        fake_prediction_loss = self.RF_loss(self.fake_prediction, self.fake_labels)
        U_errG = real_prediction_loss + fake_prediction_loss

        _, self.real_prediction = netD(real_img)
        _, self.fake_prediction = netD(fake_img)

        real_prediction_loss = self.RF_loss(self.real_prediction, self.real_labels)
        fake_prediction_loss = self.RF_loss(self.fake_prediction, self.fake_labels)


        # # # # # #
        D_Loss = real_prediction_loss + fake_prediction_loss
        D_Loss = D_Loss + U_errG
        D_Loss.backward()
        optD.step()

    def train_Gnet(self):
        self.optimizersGE.zero_grad()
        errG_total = 0

        for i in range(len(self.netsD)):

            if i == 3:
                _, fake_prediction_loss = self.netsD[i](self.fake_final_img)
                errG_1 = self.RF_loss(fake_prediction_loss, self.real_labels)

                _, fake_prediction_loss = self.netsD[i](self.fake_imgs_list[0])
                errG_2 = self.RF_loss(fake_prediction_loss, self.real_labels)

                errG_total = errG_total + errG_1 + errG_2

            if i == 0:
                errG_total = errG_total


            # # # # # # # # Regularization based on mutual information # # # # # # #

            elif i == 1:
                code_fake, _ = self.netsD[i](self.fake_mask_list[i - 1])
                code_fake = code_fake.squeeze(-1).squeeze(-1)

                errG_info = self.CE(code_fake, self.real_c)

                errG_total = errG_total + errG_info

            elif i == 2:
                code_fake, _ = self.netsD[i](self.fake_mask_inverse_list[i - 2])
                code_fake = code_fake.squeeze(-1).squeeze(-1)

                errG_info = self.CE(code_fake, self.real_c_inv)
                errG_total = errG_total + errG_info

        errG_total.backward()
        self.optimizersGE.step()

    def train_Unet(self):
        self.optimizersU.zero_grad()

        lean_mask = self.fake_imgs_list[1].detach()
        rec_loss = self.l1_loss(self.mask_fake, lean_mask)

        _, fake_prediction_loss = self.netsD[0](self.mask_fake.repeat(1,3,1,1))
        errG_1 = self.RF_loss(fake_prediction_loss, self.real_labels)

        _, fake_prediction_loss = self.netsD[0](self.mask_real.repeat(1,3,1,1))
        errG_2 = self.RF_loss(fake_prediction_loss, self.real_labels)

        errU_total = rec_loss * 2 + errG_2 + errG_1


        errU_total.backward()
        self.optimizersU.step()

    def train(self):

        self.netG, self.netsD, self.Unet= load_network()

 ############# Evaluation of unsupervised tasks ############

        if not cfg.TRAIN.FLAG:
            path = cfg.TEST.NET_U
            state_dict = torch.load(path)
            self.Unet.load_state_dict(state_dict)
            toggle_grad( self.Unet, False)
            self.Unet = self.Unet.eval()
            data_path = cfg.DATA_DIR
            trainset = datasets.CUBDataset(data_path,
                                           "train",
                                           transforms.Compose([transforms.Resize(152),
                                                               transforms.CenterCrop(128),
                                                               transforms.ToTensor(),
                                                               ]))
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=0,
                                                      drop_last=False)
            iou_s, dice_s = evaluate( self.Unet, trainloader, device)
            print("train:", iou_s , dice_s )


        self.max_epoch = cfg.TRAIN.FIRST_MAX_EPOCH
        #优化器
        self.optimizersD, self.optimizersGE, self.optimizersU = define_optimizers(self.netG, self.netsD, self.Unet)

        #self.RF_loss_un = nn.BCELoss(reduction='none')
        self.RF_loss = nn.BCELoss()
        self.CE = CrossEntropy()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.BN_2 = BinaryLoss(loss_weight = 0.5)


        avg_param_G = copy_G_params(self.netG)
        avg_param_U = copy_G_params(self.Unet)

        for self.epoch in range(cfg.TRAIN.FIRST_MAX_EPOCH):
            start_t = time.time()
            self.count = 0
            #### TRAIN ###
            for data in self.dataloader:
                self.count = self.count + 1
                self.real_img, self.real_z, self.real_p, self.real_c,  self.real_c_inv  = self.prepare_data(data)

                self.real_img = self.real_img.to(device)

                self.fake_imgs_list, self.fake_mask_list, self.fake_imgs_inverse_list, self.fake_mask_inverse_list, self.fake_final_img\
                    = self.netG(self.real_z, self.real_c, self.real_c_inv)

                self.mask_fake = self.Unet(self.fake_final_img.detach())
                self.mask_real = self.Unet(self.real_img)


                ## save_fake_img
                if self.count % 100 ==0:
                    save_img_results( None, self.mask_real.detach().cpu(), self.count, self.epoch, self.image_dir, flage=1)
                    save_img_results(None, self.real_img.detach().cpu(), self.count, self.epoch, self.image_dir,flage=0)
                    save_img_results( self.fake_imgs_list[1].detach().cpu(), None, self.count, self.epoch, self.image_dir,flage=1)
                    save_img_results( self.mask_fake.detach().cpu(), None, self.count, self.epoch, self.image_dir, flage=0)
                    save_img_results(self.fake_final_img.detach().cpu(), None, self.count, self.epoch, self.image_dir, flage=2)


                self.train_Dnet(0)

                self.train_Unet()


            end_t = time.time()
            print('''[%d/%d] Time: %.2fs
                      ''' % (self.epoch, self.max_epoch, end_t - start_t))


            data_path = cfg.DATA_DIR
            trainset = datasets.CUBDataset(data_path,
                                           "train",
                                           transforms.Compose([transforms.Resize(152),
                                                               transforms.CenterCrop(128),
                                                               transforms.ToTensor(),
                                                               ]))
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False, num_workers=0,
                                                      drop_last=True)
            iou_s, dice_s = evaluate(self.Unet, trainloader, device)

            print("train:", iou_s , dice_s )

            save_model(self.netG, self.netsD[0], self.netsD[1], self.netsD[2], self.netsD[3], self.Unet, self.epoch, self.model_dir)

            print(str(self.epoch) + 'th epoch finished')
    




if __name__ == "__main__":
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    output_dir = make_output_dir()

    trainer = Trainer(output_dir)
    print('start training now')
    trainer.train()


