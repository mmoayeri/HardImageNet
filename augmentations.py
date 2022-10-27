from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import random
'''
Implementations of common augmentations so that image and corresponding mask have the
same augmentation done to them (maintaining integrity of mask)
'''
    
def random_resized_crop(img, mask):
    transformed_imgs = []
    resize = transforms.Resize((224,224))
    i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.08,1.0),ratio=(0.75,1.33))
    coin_flip = (random.random() < 0.5)
    for x in [img, mask]:
        x = TF.crop(x, i, j, h, w)
        if coin_flip:
            x = TF.hflip(x)
        x = TF.to_tensor(resize(x))
        
        if x.shape[0] == 1:
            x = torch.cat([x, x, x], axis=0)
        
        transformed_imgs.append(x)

    img, mask = transformed_imgs
    return img, mask

def standard_resize_center_crop(img, mask, resize_shape=256):
    t = transforms.Compose([transforms.Resize((resize_shape, resize_shape)), transforms.CenterCrop(224), transforms.ToTensor()])
    # t = transforms.Compose([transforms.Resize((224,224)), transforms.CenterCrop(224), transforms.ToTensor()])
    # t = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    img, mask = [t(x) for x in [img, mask]]
    if img.shape[0] == 1:
        img = torch.cat([img, img, img], axis=0)
    if mask.shape[0] == 1:
        mask = torch.cat([mask, mask, mask], axis=0)
    return img, mask

def standard_resize_center_crop_224(img, mask):
    return standard_resize_center_crop(img, mask, 224)

def to_tens(img, mask):
    img, mask = [transforms.ToTensor()(x) for x in [img, mask]]
    return img, mask