"""Implementation of HAM."""
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

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument('--mean1', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean1.')
parser.add_argument('--std1', type=float, default=np.array([0.229, 0.224, 0.225]), help='std1.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

T_kernel = gkern(7, 3)

#channel scaling
def CS(img):
    scale = torch.ones(img.size()[0], 3, opt.image_width, opt.image_width).cuda()
    r, g, b = torch.rand(3).cuda()
    scale[:, 0] *= r
    scale[:, 1] *= g
    scale[:, 2] *= b
    scale = scale.cuda()
    return img * scale



#Spectrum masking
def SM(image, n_holes, length):
    _, c, h, w = image.shape
    mask = np.ones((c, h, w), np.float32)
    
    for l in range(c):        
        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            hole_h = length
            hole_w = length
            
            y1 = np.clip(y - hole_h // 2, 0, h)
            y2 = np.clip(y + hole_h // 2, 0, h)
            x1 = np.clip(x - hole_w // 2, 0, w)
            x2 = np.clip(x + hole_w // 2, 0, w)
            
            mask[l, y1:y2, x1:x2] = 0.
    mask = np.broadcast_to(mask,image.shape)       
    mask = torch.tensor(mask)
    mask = mask.cuda()
    return mask * image


def HAM(images, gt, model,min, max):
    """
    The attack algorithm of our proposed Spectrum Simulate Attack
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    momentum = opt.momentum
    num_iter = 10
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = images.clone()
    grad = 0
    N = 20
    for i in range(num_iter):
        noise = 0
        for n in range(N):
            
            x_dct = dct_2d(CS(x))
            x_dct = SM(x_dct,10,60)
            x_idct = idct_2d(x_dct)
            x_idct = V(x_idct, requires_grad = True)

            # DI-FGSM https://arxiv.org/abs/1803.06978
            #output = model(DI(x_idct))
            output = model(x_idct)
            loss = F.cross_entropy(output, gt)
            loss.backward()
            noise += x_idct.grad.data
        noise = noise / N

        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        #noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()

def main():

    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    for images, images_ID,  gt_cpu in tqdm(data_loader):

        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img = HAM(images, gt, model, images_min, images_max)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()