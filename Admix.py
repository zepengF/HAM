"""Implementation of Admix."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from attack_methods import DI,gkern
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from dct import *
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels

T_kernel = gkern(7, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Source Models.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument('--mean1', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean1.')
parser.add_argument('--std1', type=float, default=np.array([0.229, 0.224, 0.225]), help='std1.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--batch_size", type=int, default=8, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.backends.cudnn.benchmark = True

transforms = T.Compose(
    [T.Resize(299),T.ToTensor()]
)


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

def admix(x):
    indices = torch.arange(start=0, end=x.shape[0], dtype=torch.int32)
    return	x + 0.2 * x[torch.randperm(indices.shape[0])]
#    return torch.cat([(x + 0.2 * x[torch.randperm(indices.shape[0])]) for _ in range(3)], dim=0)


def attack(model ,x, gt):
    eps = opt.max_epsilon / 255.0
    num_iter = 10
    alpha = eps / num_iter

    decay = 1.0
    adv_images = x.clone().detach()
    momentum = torch.zeros_like(x).detach().cuda()
    y = gt
    for i in range(num_iter):
        adv_images.requires_grad = True
        noise = 0
        for j in range(3):   
            x_admix = admix(adv_images)
            for m in torch.arange(5):
                x_nes = x_admix / torch.pow(2, m)
                #DI
                #output = model(DI(x_nes))
                output = model(x_nes)                
                loss = F.cross_entropy(output, y)
                noise += torch.autograd.grad(loss, x_nes,retain_graph=False, create_graph=False)[0] / torch.pow(2, m)
        grad = noise / 15
        #TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        #grad = F.conv2d(grad, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        #MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay
        momentum = grad

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - x, min=-eps, max=eps)
        adv_images = torch.clamp(x + delta, min=0, max=1).detach()

    return adv_images

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir +'/'+ name)

def main():
    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())

    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        adv_img = attack(model , images, gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()

