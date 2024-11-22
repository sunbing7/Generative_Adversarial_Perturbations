import numpy as np
#from scipy.misc import imread, imresize, imsave
import imageio
import torch
import os, sys, time, random
import torch
import json
import numpy as np

from config import RESULT_PATH, MODEL_PATH, PROJECT_PATH, UAP_PATH, NEURON_PATH, ATTRIBUTION_PATH

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = imageio.imread(filepath)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    img = imageio.imresize(img, (256, 256))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = preprocess_img(img)
    return img


def save_img(img, filename):
    img = deprocess_img(img)
    img = img.numpy()
    img *= 255.0
    img = img.clip(0, 255)
    img = np.transpose(img, (1, 2, 0))
    img = imageio.imresize(img, (250, 200, 3))
    img = img.astype(np.uint8)
    imageio.imsave(filename, img)
    print("Image saved as {}".format(filename))


def preprocess_img(img):
    # [0,255] image to [0,1]
    min = img.min()
    max = img.max()
    img = torch.FloatTensor(img.size()).copy_(img)
    img.add_(-min).mul_(1.0 / (max - min))

    # RGB to BGR
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # [0,1] to [-1,1]
    img = img.mul_(2).add_(-1)

    # check that input is in expected range
    assert img.max() <= 1, 'badly scaled inputs'
    assert img.min() >= -1, "badly scaled inputs"

    return img


def deprocess_img(img):
    # BGR to RGB
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # [-1,1] to [0,1]
    img = img.add_(1).div_(2)

    return img

def get_model_path(dataset_name, network_arch, random_seed):
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model_path = os.path.join(MODEL_PATH, "{}_{}_{}".format(dataset_name, network_arch, random_seed))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    return model_path

