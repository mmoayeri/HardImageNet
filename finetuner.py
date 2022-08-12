import torch
import torch.nn as nn
import os
from datasets import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import timm
from torchvision import transforms, models
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_FT_ROOT = '/cmlscratch/mmoayeri/models/finetuned/' #finetune2 has the robust models w/o norm applied
_IMAGENET_ROOT = '/fs/cml-datasets/ImageNet/ILSVRC2012/'
_DEIT_ROOT = '/nfshomes/mmoayeri/.cache/torch/hub/facebookresearch_deit_main'

class FineTuner(object):
    def __init__(self, mtype='torch_resnet50', dset_name='hard_imagenet', epochs=15,
                sal_reg=False, rand_noise=False, balanced_subset=False, bg_only=False): 
        self.mtype, self.dset_name = mtype, dset_name
        self.sal_reg, self.rand_noise, self.balanced_subset = sal_reg, rand_noise, balanced_subset
        self.bg_only = bg_only
        self.init_loaders()
        self.init_model(mtype)
        self.optimizer = torch.optim.Adam(self.parameters, lr=0.0005, 
                                    betas=(0.9,0.999), weight_decay=0.0001)
        ext = '_'.join([k for x,k in zip([True, bg_only, sal_reg, rand_noise, balanced_subset], ['', 'BgOnly', 'SalReg', 'RandNoise', 'Balanced']) if x])
        self.save_path = os.path.join(_FT_ROOT, dset_name, f'{mtype}{ext}.pth')
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.best_acc = 0
        self.num_epochs = epochs
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def init_model(self, mtype):
        if 'timm_deit' in self.mtype:
            model = timm.create_model(self.mtype[len('timm_'):], pretrained=True)
            in_ftrs = model.head.in_features
            self.gradcam_layer = model.blocks[-1].norm1
            model.feat_net = nn.Sequential(model.patch_embed, model.pos_drop, model.blocks, model.norm)
            model.head = nn.Linear(in_features=in_ftrs, out_features=self.num_classes, bias=True)
            model.classifier = model.head

        elif self.mtype == 'torch_resnet50':
            net = models.resnet50(pretrained=True)
            self.gradcam_layer = net.layer4[-1]
            feat_net = nn.Sequential(*list(net.children())[:-1])
            in_ftrs = net.fc.in_features
            model = nn.Sequential()
            model.add_module('feat_net', feat_net)  
            model.add_module('flatten', nn.Flatten())
            model.add_module('classifier', nn.Linear(in_features=in_ftrs, out_features=self.num_classes, bias=True))  

        else:
            raise ValueError(f"Specified model type {self.mtype} not recognized. Update finetuner.py to facilitate model loading.")

        parameters = list(model.classifier.parameters())
        for param in model.feat_net.parameters():
            param.requires_grad = False

        self.model = model.to(device)
        self.parameters = parameters

    def init_loaders(self):
        if self.dset_name == 'hard_imagenet':
            trainset, testset = [HardImageNet(split=s, ft=True, balanced_subset=self.balanced_subset)
                                     for s in ['train', 'val']]
        elif self.dset_name == 'rival10':
            trainset, testset = [RIVAL10(split=s, ft=True) for s in ['train', 'val']]
        elif self.dset_name == 'rival20':
            trainset, testset = [RIVAL10(split=s, ft=True, twenty=True) for s in ['train', 'val']]
        self.num_classes = trainset.num_classes
        self.loaders = dict({phase:DataLoader(dset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True) 
                             for phase, dset in zip(['train', 'test'], [trainset, testset])})

    def save_model(self):
        self.model.eval()
        save_dict = dict({'state': self.model.classifier.state_dict(),
                          'acc': self.best_acc})
        torch.save(save_dict, self.save_path)
        print('\nSaved model with accuracy: {:.3f} to {}\n'.format(self.best_acc, self.save_path))

    def restore_model(self):
        print('Loading model from {}'.format(self.save_path))
        save_dict = torch.load(self.save_path)
        self.model.classifier.load_state_dict(save_dict['state'])
        self.model.eval()
        self.best_acc = save_dict['acc']

    def gradcam_layer(self):
        return self.gradcam_layer

    def process_epoch(self, phase):
        sal_pen = 0
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        correct, running_loss, total = 0, 0, 0
        for dat in tqdm(self.loaders[phase]):
            dat = [d.cuda() for d in dat]
            x, mask, y = dat

            if self.rand_noise and random.random() > 0.5:
                noise = torch.randn_like(x) * (1-mask) * 0.25
                x = torch.clamp(x + noise, 0, 1)
            
            if self.bg_only:
                x = 0.5*torch.ones_like(x) * mask + x * (1-mask)

            x = self.normalize(x)
            if self.sal_reg:
                x.requires_grad_()
                x.retain_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.sal_reg and phase == 'train':    
                non_obj_sal = x.grad * (1-mask)
                non_obj_sal.requires_grad_()
                zeros = torch.zeros(mask.shape, device=mask.device)
                saliency_pen = self.mse(non_obj_sal, zeros)
                self.optimizer.zero_grad()
                saliency_pen.backward()
                self.optimizer.step()
                sal_pen += saliency_pen.item()
                del saliency_pen, x, mask, non_obj_sal

            y_pred = logits.argmax(dim=1)
            correct += (y_pred == y).sum()
            total += y.shape[0]
            running_loss += loss.item()
        avg_loss, avg_acc = [stat/total for stat in [running_loss, correct]]
        return avg_loss, avg_acc
    
    def finetune(self):
        print('Beginning finetuning of model to be saved at {}'.format(self.save_path))

        if self.sal_reg:
            self.turn_on_grad()

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.process_epoch('train')
            test_loss, test_acc = self.process_epoch('test')
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_model()
            print('Epoch: {}/{}......Train Loss: {:.3f}......Train Acc: {:.3f}......Test Loss: {:.3f}......Test Acc: {:.3f}'
                  .format(epoch, self.num_epochs, train_loss, train_acc, test_loss, test_acc))
        print('Finetuning Complete')

    def turn_on_grad(self):
        ''' for computing gradcams '''
        for param in self.model.feat_net.parameters():
            param.requires_grad = True

if __name__ == '__main__':  
    # for sal_reg, rand_noise, balanced in [[True, True, False], [False, False, True], [True, True, True], [False, False, False]]:
    for mtype in ['timm_deit_small_patch16_224', 'torch_resnet50']:
        for dset_name in ['hard_imagenet', 'rival10', 'rival20']:
            ft = FineTuner(mtype=mtype, dset_name=dset_name, bg_only=True)
            # ft = FineTuner(mtype=mtype, dset_name=dset_name,
            #         sal_reg=sal_reg, rand_noise=rand_noise, balanced_subset=balanced)
            ft.finetune()
