import matplotlib.image as mpimg
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import pickle
import os
from PIL import Image
from augmentations import *
from hard_imagenet import _MASK_ROOT, _IMAGENET_ROOT

_ROOT = '/scratch1/mmoayeri/data/RIVAL10/'

with open(_MASK_ROOT+'meta/idx_to_wnid.pkl', 'rb') as f:
    idx_to_wnid = pickle.load(f)
wnid_to_idx = dict({v:k for k,v in idx_to_wnid.items()})

with open(_MASK_ROOT+'meta/wnid_to_rival10_id.pkl', 'rb') as f:
    wnid_to_rival10_id = pickle.load(f)

class RIVAL10(Dataset):
    def __init__(self, split='val', aug=standard_resize_center_crop, ft=False, twenty=False):
        '''
        Returns original ImageNet index when ft is False, otherwise returns label between 0 and 9
        '''
        self.aug = aug if aug is not None else (lambda x,y: (transforms.ToTensor()(x), transforms.ToTensor()(y)))
        self.split = split
        self.mask_paths = self.recover_imagenet_train_val_split()
        self.num_classes = 10 if (ft and not twenty) else 20
        self.ft = ft
        self.twenty = twenty

    def map_wnid_to_label(self, wnid):
        if self.ft:
            if self.twenty:
                return wnid_to_rival10_id[wnid][1]
            else:
                return wnid_to_rival10_id[wnid][0]
        else:
            return wnid_to_idx[wnid]

    def recover_imagenet_train_val_split(self):
        template = _ROOT+'{}/entire_object_masks/*'
        all_paths = glob.glob(template.format('test')) + glob.glob(template.format('train'))
        train, val = [], []
        for p in all_paths:
            if 'ILSVRC2012_val' in p:
                val.append(p)
            else:
                train.append(p)
        return val if self.split == 'val' else train

    def __getitem__(self, ind):
        mask_path = self.mask_paths[ind]
        mask_path_suffix = mask_path.split('/')[-1]
        wnid = mask_path_suffix.split('_')[0]
        fname = mask_path_suffix[len(wnid)+1:]

        img_path = os.path.join(_IMAGENET_ROOT, self.split, wnid, fname)
        img, mask = [Image.open(p) for p in [img_path, mask_path]]

        img, mask = self.aug(img, mask)

        class_ind = self.map_wnid_to_label(wnid)
        mask[mask > 0] = 1
        return img, mask, class_ind

    def __len__(self):
        return len(self.mask_paths)