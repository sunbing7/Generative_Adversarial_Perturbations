from __future__ import print_function
import argparse
import os
from math import log10
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from models import ResnetGenerator, weights_init
from material.models.generators import ResnetGenerator, weights_init
from data import *
import torch.backends.cudnn as cudnn
import math
import torchvision.transforms as transforms
import numpy as np
from util import get_model_path

from networks.network import *
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)

# Training settings
parser = argparse.ArgumentParser(description='generative adversarial perturbations')
parser.add_argument('--dataset', type=str,
                    default='imagenet', help='dataset')
parser.add_argument('--imagenetTrain', type=str,
                    default='/nfs01/data/imagenet-original/ILSVRC2012_img_train_caffemapping',
                    help='ImageNet train root')
parser.add_argument('--imagenetVal', type=str,
                    default='/nfs01/data/imagenet-original/ILSVRC2012_img_val_caffemapping', help='ImageNet val root')
parser.add_argument('--batchSize', type=int, default=30, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: "adam" or "sgd"')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--MaxIter', type=int, default=50, help='Iterations in each Epoch')
parser.add_argument('--MaxIterTest', type=int, default=160, help='Iterations in each Epoch')
parser.add_argument('--mag_in', type=float, default=10.0, help='l_inf magnitude of perturbation')
parser.add_argument('--expname', type=str, default='tempname', help='experiment name, output folder')
parser.add_argument('--checkpoint', type=str, default='', help='path to starting checkpoint')
parser.add_argument('--foolmodel', type=str,
                    default='resent50',
                    help='model to fool: "resent50", "incv3", "googlenet", "mobilenet", "shufflenetv2", "vgg16", or "vgg19"')
parser.add_argument('--model_in', type=str)
parser.add_argument('--mode', type=str, default='train', help='mode: "train" or "test"')
parser.add_argument('--perturbation_type', type=str, help='"universal" or "imdep" (image dependent)')
parser.add_argument('--target', type=int,
                    default=-1, help='target class: -1 if untargeted, 0..999 if targeted')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
parser.add_argument('--path_to_U_noise', type=str,
                    default='', help='path to U_input_noise.txt (only needed for universal)')
parser.add_argument('--explicit_U', type=str, default='', help='Path to a universal perturbation to use')
opt = parser.parse_args()

PRINT_DETAILS=False
PRINT_TEST_RESULT=True

if PRINT_DETAILS:
    print(opt)




#if not torch.cuda.is_available():
#    raise Exception("No GPU found.")

# train loss history
train_loss_history = []
test_loss_history = []
test_acc_history = []
test_fooling_history = []
best_fooling = 0
itr_accum = 0

# make directories
if not os.path.exists(opt.expname):
    os.mkdir(opt.expname)

if opt.perturbation_type == 'universal':
    if not os.path.exists(opt.expname + '/U_out'):
        os.mkdir(opt.expname + '/U_out')

cudnn.benchmark = True
torch.cuda.manual_seed(opt.seed)

MaxIter = opt.MaxIter
MaxIterTest = opt.MaxIterTest
gpulist = [int(i) for i in opt.gpu_ids.split(',')]
n_gpu = len(gpulist)
print('Running with n_gpu: ', n_gpu)

# define normalization means and stddevs
model_dimension = 299 if opt.foolmodel == 'incv3' else 256
center_crop = 299 if opt.foolmodel == 'incv3' else 224

mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean_arr,
                                 std=stddev_arr)

data_transform = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),
    normalize,
])

print('===> Loading datasets')

if opt.mode == 'train' and opt.dataset == 'imagenet':
    train_set = torchvision.datasets.ImageFolder(root = opt.imagenetTrain, transform = data_transform)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    _, data_test = get_data(opt.dataset)
    testing_data_loader = torch.utils.data.DataLoader(data_test,
                                                      batch_size=opt.testBatchSize,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      pin_memory=True)
else:
    data_train, data_test = get_data(opt.dataset)
    training_data_loader = torch.utils.data.DataLoader(data_test,
                                                      batch_size=opt.batchSize,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      pin_memory=True)

    testing_data_loader = torch.utils.data.DataLoader(data_test,
                                                      batch_size=opt.testBatchSize,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      pin_memory=True)


'''
test_set = torchvision.datasets.ImageFolder(root = opt.imagenetVal, transform = data_transform)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)
'''
# Init model, criterion, and optimizer

# get a path for loading the model to be attacked
# model_path = './trained_models'
model_path = get_model_path(dataset_name=opt.dataset,
                            network_arch=opt.foolmodel,
                            random_seed=123)
model_weights_path = os.path.join(model_path, opt.model_in)
num_classes, (mean_arr, stddev_arr), input_size, num_channels = get_data_specs(opt.dataset)

center_crop = input_size

target_network = get_network(opt.foolmodel,
                             input_size=input_size,
                             num_classes=num_classes,
                             finetune=False)

if opt.dataset == "caltech" or opt.dataset == 'asl':
    if 'repaired' in opt.model_in:
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
    else:
        #state dict
        orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in target_network.state_dict().items():
            if k in orig_state_dict.keys():
                new_state_dict[k] = orig_state_dict[k]

        target_network.load_state_dict(new_state_dict)

elif opt.dataset == 'eurosat':
    target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
elif opt.dataset == "imagenet" and 'repaired' in opt.model_in:
    target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))


target_network = torch.nn.DataParallel(target_network)

pretrained_clf = target_network.cuda(gpulist[0])

pretrained_clf.eval()
pretrained_clf.volatile = True

# magnitude
mag_in = opt.mag_in

print('===> Building model')

if not opt.explicit_U:
    # will use model paralellism if more than one gpu specified
    netG = ResnetGenerator(3, 3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

    # resume from checkpoint if specified
    if opt.checkpoint:
        task_label = "foolrat" if opt.target == -1 else "top1target"
        net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(opt.nEpochs) + task_label + "_{}.pth".format(
            opt.target)
        if os.path.isfile(net_g_model_out_path):
            print("=> loading checkpoint '{}'".format(net_g_model_out_path))
            netG.load_state_dict(torch.load(net_g_model_out_path, map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{}'".format(net_g_model_out_path))
        else:
            print("=> no checkpoint found at '{}'".format(net_g_model_out_path))
            netG.apply(weights_init)
    else:
        netG.apply(weights_init)

    # setup optimizer
    if opt.optimizer == 'adam':
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.optimizer == 'sgd':
        optimizerG = optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9)

    criterion_pre = nn.CrossEntropyLoss()
    criterion_pre = criterion_pre.cuda(gpulist[0])

    # fixed noise for universal perturbation
    if opt.perturbation_type == 'universal':
        noise_data = np.random.uniform(0, 255, center_crop * center_crop * 3)
        if opt.checkpoint:# and 'repaired' in opt.checkpoint:
            if opt.path_to_U_noise:
                noise_data = np.loadtxt(opt.path_to_U_noise)
                np.savetxt(opt.expname + '/U_input_noise.txt', noise_data)
            else:
                noise_data = np.loadtxt(opt.expname + '/U_input_noise.txt')
        else:
            np.savetxt(opt.expname + '/U_input_noise.txt', noise_data)
        im_noise = np.reshape(noise_data, (3, center_crop, center_crop))
        im_noise = im_noise[np.newaxis, :, :, :]
        im_noise_tr = np.tile(im_noise, (opt.batchSize, 1, 1, 1))
        noise_tr = torch.from_numpy(im_noise_tr).type(torch.FloatTensor).cuda(gpulist[0])

        im_noise_te = np.tile(im_noise, (opt.testBatchSize, 1, 1, 1))
        noise_te = torch.from_numpy(im_noise_te).type(torch.FloatTensor).cuda(gpulist[0])

def train(epoch):
    netG.train()
    global itr_accum
    global optimizerG

    for itr, (image, _) in enumerate(training_data_loader, 1):
        if itr > MaxIter:
            break
        if itr == 34:
            tst = 1
        if opt.target == -1:
            # least likely class in nontargeted case
            pretrained_label_float = pretrained_clf(image.cuda(gpulist[0]))
            _, target_label = torch.min(pretrained_label_float, 1)
        else:
            # targeted case
            target_label = torch.LongTensor(image.size(0))
            target_label.fill_(opt.target)
            target_label = target_label.cuda(gpulist[0])

        itr_accum += 1
        if opt.optimizer == 'sgd':
            lr_mult = (itr_accum // 1000) + 1
            optimizerG = optim.SGD(netG.parameters(), lr=opt.lr/lr_mult, momentum=0.9)

        image = image.cuda(gpulist[0])

        ## generate per image perturbation from fixed noise
        if opt.perturbation_type == 'universal':
            delta_im = netG(noise_tr)
        else:
            delta_im = netG(image)

        delta_im = normalize_and_scale(delta_im, 'train')
        delta_im = delta_im[:len(image)]
        netG.zero_grad()

        recons = torch.add(image.cuda(gpulist[0]), delta_im.cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image[:,cii,:,:].min(), image[:,cii,:,:].max())

        output_pretrained = pretrained_clf(recons.cuda(gpulist[0]))

        # attempt to get closer to least likely class, or target
        loss = torch.log(criterion_pre(output_pretrained, target_label))

        loss.backward()
        optimizerG.step()

        train_loss_history.append(loss.item())
        #if PRINT_DETAILS:
        #    print("===> Epoch[{}]({}/{}) loss: {:.4f}".format(epoch, itr, len(training_data_loader), loss.item()))


def test():
    if not opt.explicit_U:
        netG.eval()
    correct_recon = 0
    correct_orig = 0
    fooled = 0
    total = 0

    if opt.perturbation_type == 'universal':
        if opt.explicit_U:
            U_loaded = torch.load(opt.explicit_U)
            U_loaded = U_loaded.expand(opt.testBatchSize, U_loaded.size(1), U_loaded.size(2), U_loaded.size(3))
            delta_im = normalize_and_scale(U_loaded, 'test')
        else:
            delta_im = netG(noise_te)
            delta_im = normalize_and_scale(delta_im, 'test')

    for itr, (image, class_label) in enumerate(testing_data_loader):
        if itr > MaxIterTest:
            break

        image = image.cuda(gpulist[0])

        if opt.perturbation_type == 'imdep':
            delta_im = netG(image)
            delta_im = normalize_and_scale(delta_im, 'test')
        delta_im = delta_im[:len(image)]
        recons = torch.add(image.cuda(gpulist[0]), delta_im[0:image.size(0)].cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image[:,cii,:,:].min(), image[:,cii,:,:].max())

        outputs_recon = pretrained_clf(recons.cuda(gpulist[0]))
        outputs_orig = pretrained_clf(image.cuda(gpulist[0]))
        _, predicted_recon = torch.max(outputs_recon, 1)
        _, predicted_orig = torch.max(outputs_orig, 1)
        total += image.size(0)
        correct_recon += (predicted_recon == class_label.cuda(gpulist[0])).sum()
        correct_orig += (predicted_orig == class_label.cuda(gpulist[0])).sum()

        if opt.target == -1:
            fooled += (predicted_recon != predicted_orig).sum()
        else:
            fooled += (predicted_recon == opt.target).sum()

        if itr % 50 == 1 and PRINT_DETAILS:
            print('Images evaluated:', (itr*opt.testBatchSize))
            # undo normalize image color channels
            delta_im_temp = torch.zeros(delta_im.size())
            for c2 in range(3):
                recons[:,c2,:,:] = (recons[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                image[:,c2,:,:] = (image[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                delta_im_temp[:,c2,:,:] = (delta_im[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
            if not os.path.exists(opt.expname):
                os.mkdir(opt.expname)

            post_l_inf = (recons - image[0:recons.size(0)]).abs().max() * 255.0
            print("Specified l_inf:", mag_in, "| maximum l_inf of generated perturbations: %.2f" % (post_l_inf.item()))

            torchvision.utils.save_image(recons, opt.expname+'/reconstructed_{}.png'.format(itr))
            torchvision.utils.save_image(image, opt.expname+'/original_{}.png'.format(itr))
            torchvision.utils.save_image(delta_im_temp, opt.expname+'/delta_im_{}.png'.format(itr))
            print('Saved images.')

    test_acc_history.append((100.0 * correct_recon / total))
    test_fooling_history.append((100.0 * fooled / total))
    if PRINT_TEST_RESULT:
        print('Accuracy of the pretrained network on reconstructed images: %.2f%%' % (100.0 * float(correct_recon) / float(total)))
        print('Accuracy of the pretrained network on original images: %.2f%%' % (100.0 * float(correct_orig) / float(total)))
    if opt.target == -1:
        print('Fooling ratio: %.2f%%' % (100.0 * float(fooled) / float(total)))
    else:
        print('Top-1 Target Accuracy: %.2f%%' % (100.0 * float(fooled) / float(total)))

def normalize_and_scale(delta_im, mode='train'):
    if opt.foolmodel == 'incv3':
        delta_im = nn.ConstantPad2d((0,-1,-1,0),0)(delta_im) # crop slightly to match inception

    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5 # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    bs = opt.batchSize if (mode == 'train') else opt.testBatchSize
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
            gpu_id = gpulist[1] if n_gpu > 1 else gpulist[0]
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im


def checkpoint_dict(epoch):
    netG.eval()
    global best_fooling
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    task_label = "foolrat" if opt.target == -1 else "top1target"

    #net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(test_fooling_history[epoch-1])
    net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(opt.target)
    if opt.perturbation_type == 'universal':
        u_out_path = opt.expname + "/U_out/U_epoch_{}_".format(epoch) + task_label + "_{}.pth".format(test_fooling_history[epoch-1])
    if test_fooling_history[epoch-1] > best_fooling:
        best_fooling = test_fooling_history[epoch-1]
        torch.save(netG.state_dict(), net_g_model_out_path)
        if opt.perturbation_type == 'universal':
            torch.save(netG(noise_te[0:1]), u_out_path)
        print("Checkpoint saved to {}".format(net_g_model_out_path))
    else:
        print("No improvement:", test_fooling_history[epoch-1], "Best:", best_fooling)


def print_history():
    # plot history for training loss
    if opt.mode == 'train':
        plt.plot(train_loss_history)
        plt.title('Model Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.legend(['Training Loss'], loc='upper right')
        plt.savefig(opt.expname+'/reconstructed_loss_'+opt.mode+'.png')
        plt.clf()

    # plot history for classification testing accuracy and fooling ratio
    plt.plot(test_acc_history)
    plt.title('Model Testing Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Testing Classification Accuracy'], loc='upper right')
    plt.savefig(opt.expname+'/reconstructed_acc_'+opt.mode+'.png')
    plt.clf()

    plt.plot(test_fooling_history)
    plt.title('Model Testing Fooling Ratio')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Epoch')
    plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.savefig(opt.expname+'/reconstructed_foolrat_'+opt.mode+'.png')
    print("Saved plots.")

if opt.mode == 'train':
    for epoch in range(1, opt.nEpochs + 1):
        PRINT_TEST_RESULT = False
        train(epoch)
        if epoch == opt.nEpochs:
            print('Testing....')
            PRINT_TEST_RESULT = True
        test()
        checkpoint_dict(epoch)
    #print_history()
elif opt.mode == 'test':
    print('Testing...')
    PRINT_TEST_RESULT = True
    test()
    #print_history()
