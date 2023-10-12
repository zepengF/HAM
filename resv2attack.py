# encoding:utf-8
"""Implementation of sample attack."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from torch.autograd import Variable as V
# from torch.autograd.gradcheck import zero_gradients
from torch.utils import data
from dct import *
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from attack_methods import DI,gkern

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


list_nets = [
    'tf_inception_v3',
    'tf_inception_v4',
    'tf_resnet_v2_50',
    'tf_resnet_v2_101',
    'tf_resnet_v2_152',
    'tf_inc_res_v2',
    'tf_adv_inception_v3',
    'tf_ens3_adv_inc_v3',
    'tf_ens4_adv_inc_v3',
    'tf_ens_adv_inc_res_v2'
    ]

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='The ID of GPU to use.')
parser.add_argument('--input_csv', type=str, default='dataset/images.csv', help='Input csv with images.')
parser.add_argument('--input_dir', type=str, default='dataset/images/', help='Input images.')
parser.add_argument('--output_dir', type=str, default='outputs/res152_ssa/', help='Output directory with adv images.')
parser.add_argument('--model_dir', type=str, default='models/', help='Model weight directory.')
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument('--white_model', type=str, default='tf_resnet_v2_152', help='Substitution model.')

parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations")
parser.add_argument("--rho", type=float, default=0.5, help="Tuning factor")
parser.add_argument("--sigma", type=float, default=16.0, help="Std of random noise")

opt = parser.parse_args()
device = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def mkdir(path):
    """Check if the folder exists, if it does not exist, create it"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


class Normalize(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        (input - mean) / std
        ImageNet normalize:
            'tensorflow': mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            'torch': mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
            
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class ImageNet(data.Dataset):
    """load data from img and csv"""
    def __init__(self, dir, csv_path, transforms=None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel']
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)


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
    
    if 'inc' in net_name:
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), 
            net.KitModel(model_path).eval().cuda(),)
    else:
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), 
            net.KitModel(model_path).eval().cuda(),)
    return model


def get_models(list_nets, model_dir):
    """load models with dict"""
    nets = {}
    for net in list_nets:
        nets[net] = get_model(net, model_dir)
    return nets

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def save_img(images, filenames, output_dir):
    """save high quality jpeg"""
    mkdir(output_dir)
    for i, filename in enumerate(filenames):
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(os.path.join(output_dir, filename))
T_kernel = gkern(7, 3)





def attack(model, img, label,min,max):
    """generate adversarial images"""
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    image_width = opt.image_width
    momentum = opt.momentum
    N = 20
    x = img.clone()
    grad = 0
    for i in range(num_iter):
        noise = 0
        for n in range(N):          
            gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (16.0 / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * 0.5 + 1 - 0.5).cuda()
            x_idct = idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad = True)
            output = model(x_idct)
            #output = model(DI(x_idct))
            loss = F.cross_entropy(output[0], label)
            loss.backward()
            noise += x_idct.grad.data
        noise = noise / N

        #TI-FGSM
        #noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
        # MI-FGSM
        noise = noise / torch.abs(noise).mean([1,2,3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def main():
    transforms = T.Compose([T.ToTensor()])

    # Load inputs
    inputs = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(inputs, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    input_num = len(inputs)

    # Create models
    models = get_models(list_nets, opt.model_dir)

    # Initialization parameters
    correct_num = {}
    logits = {}
    for net in list_nets:
        correct_num[net] = 0

    # Start iteration
    for images, filename, label in tqdm(data_loader):
        label = label.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
        # Start Attack
        adv_img = attack(models[opt.white_model], images, label, images_min, images_max)

        # Save adversarial examples
        save_img(adv_img, filename, opt.output_dir)
        '''
        # Prediction
        with torch.no_grad():
            for net in list_nets:
                if "inc" in net:
                    logits[net] = models[net](adv_img)[0]
                else:
                    logits[net] = models[net](adv_img)
                correct_num[net] += (torch.argmax(logits[net], axis=1) != label).detach().sum().cpu()

    # Print attack success rate
    for net in list_nets:
        print('{} attack success rate: {:.2%}'.format(net, correct_num[net]/input_num))
        '''

if __name__ == '__main__':
    seed_torch(0)
    main()
