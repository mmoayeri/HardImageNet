import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from tqdm import tqdm
from binascii import a2b_base64
import numpy as np
import pickle
from PIL import Image
import os
from datasets import *
import timm
from torchvision import transforms, models
from finetuner import *

def save_uri_as_img(uri, fpath='tmp.png', remove_fourth_channel_and_binarize=False):
    ''' saves raw mask and returns it as an image'''
    binary_data = a2b_base64(uri)
    with open(fpath, 'wb') as f:
        f.write(binary_data)
    img = mpimg.imread(fpath)
    if remove_fourth_channel_and_binarize:
        img = img[:,:,:3]
        img = binarize(img)
        Image.fromarray(np.uint8(255*img)).save(fpath)
    return img

def recolor_mask(img, color=[20, 21, 110]):
    color = [x/255 for x in color]
    on_pixels = np.where(img[:,:,0]!=0)
    img[on_pixels[0], on_pixels[1], 0:3] = color
    return img

def read_img_from_url(url):
    return mpimg.imread(requests.get(url, stream=True).raw, format='jpeg')

def overlay_and_save_img(og, mask, overlay_fpath):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(og)
    mask = recolor_mask(mask)
    ax.imshow(mask, alpha=0.65)
    fig.tight_layout()
    fig.savefig(overlay_fpath)
    plt.close()

def change_2d_to_3d(img):
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    return img

def binarize(mask):
    height, width = [mask.shape[x] for x in [0,1]]
    on_pixels = np.where(mask!=0)
    mask2d = np.zeros((height, width))
    mask2d[on_pixels[0], on_pixels[1]] = 1
    return mask2d

def intersection_over_union(masks):
    # masks = [binarize(m) for m in masks]
    intersection = masks[0]
    for mask in masks[1:]:
        intersection = intersection * mask
    intersection = np.sum(intersection != 0)
    union = np.sum(np.sum(masks, axis=0) != 0)
    return intersection / union#, intersection, union

def load_meta_files(fname):
    with open(f'/scratch1/mmoayeri/hard_imagenet/data_collection/meta/{fname}.pkl', 'rb') as f:
        dat = pickle.load(f)
    return dat 

def cache_results(results_key, results):
    with open(f'./results/{results_key}.pkl', 'wb') as f:
        pickle.dump(results, f)

def load_cached_results(results_key):
    if os.path.exists(f'./results/{results_key}.pkl'):
        with open(f'./results/{results_key}.pkl', 'rb') as f:
            d = pickle.load(f)
    else:
        d = dict()
    return d

def get_dset(dset_name, ft, bs=32, split='val'):
    if dset_name == 'hard_imagenet':
        dset = HardImageNet(ft=ft, split=split)
    elif dset_name == 'rival10':
        dset = RIVAL10(ft=ft, split=split)
    elif dset_name == 'rival20':
        dset = RIVAL10(ft=ft, twenty=True, split=split)
    return dset

def get_model(mkey, dset_name, ft):
    if not ft:
        if mkey == 'torch_resnet50':
            model = models.resnet50(pretrained=True).eval().cuda()
            target_layer = model.layer4[-1]
        elif mkey == 'timm_deit_small_patch16_224':
            model = timm.create_model(mkey[len('timm_'):], pretrained=True).eval().cuda()
            target_layer = model.blocks[-1].norm1
    else:
        finetuner = FineTuner(dset_name=dset_name, mtype=mkey); finetuner.restore_model(); finetuner.turn_on_grad()
        model = finetuner.model
        target_layer = finetuner.gradcam_layer
    return model, target_layer 