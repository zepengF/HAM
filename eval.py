"""Implementation of evaluate attack result."""
import os
import torch
import numpy as np
import pretrainedmodels
from torch.autograd import Variable as V
from torch import nn
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from loader import ImageNet
from torch.utils.data import DataLoader

from torchvision.models.vgg import vgg16
from torchvision.models.resnet import resnet50,resnet101,resnet152

batch_size = 10

input_csv = './dataset/images.csv'
input_dir = './dataset/images'
adv_dir = './outputs'


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

norm1 = T.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
resize1 = T.Resize(224)

norm2 = T.Normalize(np.array([0.5,0.5,0.5]), np.array([0.5,0.5,0.5]))
resize2 = T.Resize(299)

def get_model(net_name):
    if net_name == 'resnet50':
        model = nn.Sequential(norm1,resize1,
                            pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').eval().cuda())
    if net_name == 'resnet101':
        model = nn.Sequential(norm1,resize1,
                            pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet').eval().cuda())
    if net_name == 'resnet152':
        model = nn.Sequential(norm1,resize1,
                            pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet').eval().cuda())        
    if net_name == 'inception_v3':
        model = nn.Sequential(norm2,resize2,
                            pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    if net_name == 'inception_v4':
        model = nn.Sequential(norm2,resize2,
                            pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())
    if net_name == 'inception_resnet_v2':
        model = nn.Sequential(norm2,resize2,
                            pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet').eval().cuda())   
    if net_name == 'vgg16':
        model = nn.Sequential(norm1,resize1,
                            pretrainedmodels.vgg16(num_classes=1000, pretrained='imagenet').eval().cuda())
    if net_name == 'vgg19':
        model = nn.Sequential(norm1,resize1,
                            pretrainedmodels.vgg19(num_classes=1000, pretrained='imagenet').eval().cuda())      
    return model

def verify(model_name):

    model = get_model(model_name)

    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            pred = torch.argmax(model(images), dim=1).view(1,-1)
            sum += ((gt != pred.squeeze(0)).sum().item())
    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))
    return sum / 1000.0

def main():

    model_names = ['inception_v3','inception_v4','inception_resnet_v2','resnet50','resnet101''resnet152','vgg16','vgg19']
    avg = 0
    for model_name in model_names:
         avg += verify(model_name)
         print("===================================================")
    print('avg acc = {:.2%}'.format(avg / len(model_names)))
if __name__ == '__main__':
    main()