##############
# This is first going
# work hard!!
# best wish
#################

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
from evals import *
from inception import InceptionV3
import random
from utils import *
from itertools import chain
from copy import deepcopy
from tensorboardX import summary
from tensorboardX import FileWriter
import torchvision.transforms as transforms
cudnn.benchmark = True

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

def define_optimizers(netG, netsD):
    # define optimizer for G and D

    optimizersD = []
    for i in range(len(netsD)):
        if i == 3 or i== 0:
            optimizersD.append(optim.Adam(netsD[i].parameters(), lr=2e-4, betas=(0.5, 0.999)))
        else:
            optimizersD.append(None)

    params = chain(netG.parameters(), netsD[0].parameters(), netsD[1].parameters(), netsD[2].parameters())

    optimizerGE = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))



    return optimizersD, optimizerGE


def load_network():
    # prepare G net
    netG = G_NET_of_DSComGAN().to(device)
    netG = torch.nn.DataParallel(netG,  device_ids=gpus)


    netsD = [ MASK_D(), FORE_D(), BACK_D() , IMAGE_D()]
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)

    return netG, netsD

def save_model(netG, D0, D1, D2, D3,   epoch, model_dir):
    torch.save(netG.state_dict(), '%s/G_%d.pth' % (model_dir, epoch))
    torch.save(D0.state_dict(), '%s/D0_%d.pth' % (model_dir, epoch))
    torch.save(D1.state_dict(), '%s/D1_%d.pth' % (model_dir, epoch))
    torch.save(D2.state_dict(), '%s/D2_%d.pth' % (model_dir, epoch))
    torch.save(D3.state_dict(), '%s/D3_%d.pth' % (model_dir, epoch))
    #torch.save(Unet.state_dict(), '%s/Unet_%d.pth' % (model_dir, epoch))



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
        
        
    def prepare_code_RA(self):
        rand_z = torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM).normal_(0, 1).to(device) * 4  
        rand_c = torch.zeros(self.batch_size, cfg.FINE_GRAINED_CATEGORIES).to(device)   
        rand_idx = [i for i in range(cfg.FINE_GRAINED_CATEGORIESs)]
        random.shuffle(rand_idx)   
        for i, idx in enumerate(rand_idx[:self.batch_size]):
            rand_c[i, idx] = 1
        rand_c_inv = rand_c
        
        return rand_z, rand_c, rand_c_inv

     

    def prepare_data(self, data):

        _, real_img, real_c, _, _ = data

        real_img = real_img.to(device)
        real_z = torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM).normal_(0, 1).to(device) * 4
        real_c = real_c.to(device)
        real_c_inv = real_c

        return real_img, real_z, real_c, real_c_inv

    def train_Dnet(self, idx):

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        netD.zero_grad()
        real_img = self.real_img
        fake_img = self.fake_final_img.detach()

        # # # # # # # # for extracting global features  # # # # # # #
        fake_img_1 = self.fake_imgs_list[0].detach()
        _, self.real_prediction = netD(real_img)
        _, self.fake_prediction = netD(fake_img_1)
        real_prediction_loss = self.RF_loss(self.real_prediction, self.real_labels)
        fake_prediction_loss = self.RF_loss(self.fake_prediction, self.fake_labels)
        Ex_errG = real_prediction_loss + fake_prediction_loss

        _, self.real_prediction = netD(real_img)
        _, self.fake_prediction = netD(fake_img)

        real_prediction_loss = self.RF_loss(self.real_prediction, self.real_labels)
        fake_prediction_loss = self.RF_loss(self.fake_prediction, self.fake_labels)

        # # # # # # # # Adversarial training  # # # # # # #

        D_Loss = real_prediction_loss + fake_prediction_loss
        D_Loss = D_Loss + Ex_errG
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


    def train(self):

        self.netG, self.netsD= load_network()

############# Evaluation of image generation  ###############

        if not cfg.TRAIN.FLAG:
            path = cfg.TEST.NET_G
            state_dict = torch.load(path)
            self.netG.load_state_dict(state_dict)
            toggle_grad(self.netG, False)
            netG = self.netG.eval()
            temp = 0
            BATCH = 100
            for bg in range(0, 200):
                bg_code = torch.zeros([BATCH, cfg.FINE_GRAINED_CATEGORIES])
                bg_code[:, bg] = 1
                c_code = bg_code
                noise = torch.FloatTensor(BATCH, 100).normal_(0, 1).to(device) * 4
                fake_imgs_list, fake_mask_list, fake_imgs_inverse_list, fake_mask_inverse_list, fake_final_img \
                    = netG(noise, c_code, bg_code)
                for i in range(0, BATCH):
                    temp = temp + 1
                    save_img_results(None, fake_final_img[i].detach().cpu(), temp, bg, self.image_dir, flage=0)
            print('20,000 images have been generated!')
            dataset = ImageDataset(self.image_dir, exts=['png', 'jpg'])
            ## get inception score on 20,000 samples
            print(inception_score(dataset, cuda=True, batch_size=32, resize=True, splits=10))
            
            ## get fid score on randomly generated samples
            self.epoch = 0
            print('Get the statistic of training images for computing fid score.')
            self.inception = InceptionV3([3]).to(device)
            self.inception.eval()
            pred_arr = np.empty((len(self.dataset), 2048))
            start_idx = 0
            for data in self.dataloader:
                batch = data[1].to(device)
                with torch.no_grad():
                    pred = self.inception(batch)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = nn.AdaptiveAvgPool2d(pred, output_size=(1, 1))
                    # adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                pred_arr[start_idx:start_idx + pred.shape[0]] = pred
                start_idx = start_idx + pred.shape[0]
            self.mu = np.mean(pred_arr, axis=0)
            self.sig = np.cov(pred_arr, rowvar=False)
            pred_arr = np.empty((len(self.dataset), 2048))
            start_idx = 0
            for i in range(len(self.dataset) // self.batch_size):
                real_z, real_c, real_c_inv = self.prepare_code_RA()
                with torch.no_grad():
                    _, _, _, _, fake_final_img = netG(real_z, real_c, real_c_inv)
                    pred = self.inception(fake_final_img)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    print('size mismatch error occurred during the fid score computation!')
                    pred = nn.AdaptiveAvgPool2d(pred, output_size=(1, 1))
                    # adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                pred_arr[start_idx:start_idx + pred.shape[0]] = pred
                start_idx = start_idx + pred.shape[0]
            cur_mu = np.mean(pred_arr, axis=0)
            cur_sig = np.cov(pred_arr, rowvar=False)
            cur_fid = calculate_frechet_distance(self.mu, self.sig, cur_mu, cur_sig)
            print(str(self.epoch) + "th epoch finished", "\t fid : ", "{:.3f}".format(cur_fid))
            
############# Evaluation Completed  ###############

        self.max_epoch = cfg.TRAIN.FIRST_MAX_EPOCH

        self.optimizersD, self.optimizersGE= define_optimizers(self.netG, self.netsD)

        self.RF_loss = nn.BCELoss()
        self.CE = CrossEntropy()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.BN_2 = BinaryLoss(loss_weight = 1)

        avg_param_G = copy_G_params(self.netG)

        for self.epoch in range(cfg.TRAIN.FIRST_MAX_EPOCH):
            start_t = time.time()
            self.count = 0
            #### TRAIN ###
            for data in self.dataloader:
                self.count = self.count + 1
                self.real_img, self.real_z, self.real_c,  self.real_c_inv  = self.prepare_data(data)

                self.real_img = self.real_img.to(device)

                self.fake_imgs_list, self.fake_mask_list, self.fake_imgs_inverse_list, self.fake_mask_inverse_list, self.fake_final_img\
                    = self.netG(self.real_z, self.real_c, self.real_c_inv)

                ## save_fake_img
                if self.count % 50 ==0:
                    save_img_results(None, self.fake_final_img.detach().cpu(), self.count, self.epoch, self.image_dir, flage=0)
                    save_img_results(None, self.fake_imgs_list[1].detach().cpu(), self.count, self.epoch, self.image_dir, flage=1)


                self.train_Dnet(3)

                self.train_Gnet()

                for avg_p, p in zip(avg_param_G, self.netG.parameters()):
                    avg_p.mul_(0.999).add_(0.001, p.data)
            end_t = time.time()
            print('''[%d/%d] Time: %.2fs
                      ''' % (self.epoch, self.max_epoch, end_t - start_t))

            backup_para = copy_G_params(self.netG)
            load_params(self.netG, avg_param_G)

            save_model(self.netG, self.netsD[0], self.netsD[1], self.netsD[2], self.netsD[3], self.epoch, self.model_dir)

            load_params(self.netG, backup_para)

            print(str(self.epoch) + 'th epoch finished')















if __name__ == "__main__":
    manualSeed = random.randint(1, 10000)
    print(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    output_dir = make_output_dir()

    trainer = Trainer(output_dir)
    print('start training now')
    trainer.train()


