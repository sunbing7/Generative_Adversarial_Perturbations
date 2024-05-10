from os.path import join
import os
from dataset import DatasetFromFolder
import torchvision.transforms as transforms
from config import *
import torchvision.datasets as dset
import numpy as np
import torch


def get_training_set(root_dir):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir)


#from my project
def get_data_specs(pretrained_dataset):
    if pretrained_dataset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000
        input_size = 224
        # input_size = 299 # inception_v3
        num_channels = 3
    elif pretrained_dataset == "imagenet_caffe":
        mean = [123 / 255, 117 / 255, 104 / 255]
        std = [1 / 255, 1 / 255, 1 / 255]
        num_classes = 1000
        input_size = 224
        # input_size = 299 # inception_v3
        num_channels = 3
    elif pretrained_dataset == "cifar10":
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 10
        input_size = 224#32
        num_channels = 3
    elif pretrained_dataset == "cifar100":
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 100
        input_size = 32
        num_channels = 3
    elif pretrained_dataset == 'caltech':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 101
        input_size = 224
        num_channels = 3
    elif pretrained_dataset == 'asl':
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 29
        input_size = 200
        num_channels = 3
    else:
        raise ValueError
    return num_classes, (mean, std), input_size, num_channels


def get_data(dataset):
    num_classes, (mean, std), input_size, num_channels = get_data_specs(dataset)

    if dataset == "imagenet":
        # use imagenet 2012 validation set as uap training set
        # use imagenet DEV 1000 sample dataset as the test set
        # traindir = os.path.join(IMAGENET_PATH, 'train')
        # valdir = os.path.join(IMAGENET_PATH, 'val')
        traindir = os.path.join(IMAGENET_PATH, 'validation')
        # traindir = IMAGENET_PATH
        # valdir = os.path.join(IMAGENET_PATH, 'ImageNet1k')

        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        full_val = dset.ImageFolder(root=traindir, transform=train_transform)
        full_val = fix_labels(full_val)

        full_index = np.arange(0, len(full_val))
        index_test = np.load(IMAGENET_PATH + '/validation/index_test.npy').astype(np.int64)
        index_train = [x for x in full_index if x not in index_test]
        train_data = torch.utils.data.Subset(full_val, index_train)
        test_data = torch.utils.data.Subset(full_val, index_test)
        #print('test size {} train size {}'.format(len(test_data), len(train_data)))

    elif dataset == "coco":
        train_transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.143)),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.143)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_data = dset.CocoDetection(root=COCO_2017_TRAIN_IMGS,
                                        annFile=COCO_2017_TRAIN_ANN,
                                        transform=train_transform)
        test_data = dset.CocoDetection(root=COCO_2017_VAL_IMGS,
                                       annFile=COCO_2017_VAL_ANN,
                                       transform=test_transform)

    elif dataset == "voc":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(input_size * 1.143)),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(input_size * 1.143)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_data = VOCDetection(root=VOC_2012_ROOT,
                                  year="2012",
                                  image_set='train',
                                  transform=train_transform)
        test_data = VOCDetection(root=VOC_2012_ROOT,
                                 year="2012",
                                 image_set='val',
                                 transform=test_transform)

    elif dataset == "places365":
        traindir = os.path.join(PLACES365_ROOT, "train")
        testdir = os.path.join(PLACES365_ROOT, "train")
        # Places365 downloaded as 224x224 images

        train_transform = transforms.Compose([
            transforms.Resize(input_size),  # Places images downloaded as 224
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data = dset.ImageFolder(root=testdir, transform=test_transform)
    elif dataset == 'caltech':
        traindir = os.path.join(CALTECH_PATH, "train")
        testdir = os.path.join(CALTECH_PATH, "test")

        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_data_full = dset.ImageFolder(root=traindir, transform=train_transform)
        train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                                                               size=int(0.05 * len(train_data_full)),
                                                                               replace=False))
        test_data = dset.ImageFolder(root=testdir, transform=test_transform)
    elif dataset == 'asl':
        traindir = os.path.join(ASL_PATH, "train")
        testdir = os.path.join(ASL_PATH, "test")

        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_data_full = dset.ImageFolder(root=traindir, transform=train_transform)
        train_data = torch.utils.data.Subset(train_data_full, np.random.choice(len(train_data_full),
                                                                               size=int(0.05 * len(train_data_full)),
                                                                               replace=False))
        test_data = dset.ImageFolder(root=testdir, transform=test_transform)
    return train_data, test_data


def fix_labels(test_set):
    val_dict = {}
    groudtruth = os.path.join(IMAGENET_PATH, 'validation/classes.txt')

    i = 0
    with open(groudtruth) as file:
        for line in file:
            (key, class_name) = line.split(':')
            val_dict[key] = i
            i = i + 1

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        class_id = test_set.samples[i][0].split('/')[-1].split('.')[0].split('_')[-1]
        org_label = val_dict[class_id]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set
