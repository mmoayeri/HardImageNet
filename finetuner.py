import torch
import torch.nn as nn
import os
from datasets import *
from torch.utils.data import DataLoader
from tqdm import tqdm
# from robustness.datasets import ImageNet
# from robustness.model_utils import make_and_restore_model
import argparse
import timm
from torchvision import transforms, models
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_FT_ROOT = '/scratch1/mmoayeri/models/finetuned2/' #finetune2 has the robust models w/o norm applied
_IMAGENET_ROOT = '/scratch1/shared/datasets/ILSVRC2012/'
_DEIT_ROOT = '/nfshomes/mmoayeri/.cache/torch/hub/facebookresearch_deit_main'

class FineTuner(object):
    def __init__(self, mtype='torch_resnet50', dset_name='hard_imagenet', epochs=15,
                sal_reg=False, rand_noise=False, balanced_subset=False): 
        self.mtype, self.dset_name = mtype, dset_name
        self.sal_reg, self.rand_noise, self.balanced_subset = sal_reg, rand_noise, balanced_subset
        self.init_loaders()
        self.init_model(mtype)
        self.optimizer = torch.optim.Adam(self.parameters, lr=0.0005, 
                                    betas=(0.9,0.999), weight_decay=0.0001)
        ext = '_'.join([k for x,k in zip([True, sal_reg, rand_noise, balanced_subset], ['', 'SalReg', 'RandNoise', 'Balanced']) if x])
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

        # elif self.mtype == 'simclr':

        #     class SimCLRWrapper(nn.Module):
        #         def __init__(self):
        #             super(SimCLRWrapper, self).__init__()
        #             from pl_bolts.models.self_supervised import SimCLR
        #             weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        #             simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
        #             # simclr.freeze()
        #             self.model = simclr.encoder
        #             nftrs = self.model.fc.in_features
        #             self.model.fc = nn.Linear(in_features=nftrs, out_features=self.num_classes, bias=True)
        #             # print(self.model)
                
        #         def forward(self, x):
        #             ftrs = self.model(x)[0]
        #             logits = self.model.fc(ftrs)
        #             return logits

        #     # match our notation later
        #     model = SimCLRWrapper()
        #     model.feat_net = nn.Sequential(*list(model.model.children())[:-1])
        #     model.classifier = model.model.fc
        #     self.gradcam_layer = model.model.layer4[-1]

        # elif 'clip' in mtype:
        #     assert ('ViT' in mtype or 'RN' in mtype), 'CLIP is only supported on ViT-B/16, ViT-B/32, RN50'
        #     import clip
        #     clip_mtype = mtype.split('clip_')[-1]
        #     if 'ViT' in clip_mtype:
        #         clip_mtype = 'ViT-B/'+clip_mtype[-2:]

        #     class CLIPWrapper(nn.Module):
        #         def __init__(self, mtype):
        #             super(CLIPWrapper, self).__init__()
        #             self.feat_net, self.preprocess = clip.load(mtype, device='cuda')
        #             in_ftrs = self.feat_net.encode_image(torch.rand(5,3,224,224).cuda()).shape[1]
        #             # in_ftrs =  512 if 'ViT' in mtype else 1024
        #             self.classifier = nn.Linear(in_features=in_ftrs, out_features=10, bias=True)

        #         def forward(self, x):
        #             # img_ftrs = self.feat_net.encode_image(self.preprocess(x))
        #             img_ftrs = self.feat_net.encode_image(x).float()
        #             logits = self.classifier(img_ftrs)
        #             return logits

        #     model = CLIPWrapper(clip_mtype)
        #     if 'ViT' in clip_mtype:
        #         self.gradcam_layer = model.feat_net.visual.transformer.resblocks[-1].ln_1
        #     elif 'RN' in clip_mtype:
        #         self.gradcam_layer = model.feat_net.visual.layer4[-1]

        # elif 'robust' in self.mtype:
        #     if 'eps' in self.mtype:
        #         eps = self.mtype.split('eps')[-1]
        #     else:
        #         eps = 3
        #     # arch = 'resnet50' if '50' in self.mtype else 'resnet18'
        #     ds_ = ImageNet(_IMAGENET_ROOT)
        #     mkey = self.mtype[len('robust_'):]
        #     if 'resnet' in mkey and 'wide' not in mkey:
        #         arch = mkey.split('_')[0]
        #         add_custom_forward=False
        #     else:
        #         arch = get_arch(mkey[:mkey.index('_l2')])
        #         add_custom_forward=True
        #     add_custom_forward = False if 'wide' in mkey else add_custom_forward
        #     # arch = models.wide_resnet50_2() if 'wide' in self.mtype else self.mtype.split('_')[1]
        #     checkpoint_fp = "/cmlscratch/mmoayeri/dcr_models/pretrained-robust/{}.ckpt".format(mkey)
        #     model, _ = make_and_restore_model(arch=arch, dataset=ds_,
        #                 resume_path=checkpoint_fp, add_custom_forward=add_custom_forward)
        #     self.gradcam_layer = model.model.layer4[-1] if 'resnet' in mkey else None
        #     feat_net = nn.Sequential(*list(model.model.model.children())[:-1])
        #     model = nn.Sequential()
        #     model.add_module('feat_net', feat_net)
        #     out = model(torch.zeros(5,3,224,224).to(device))
        #     model.add_module('flatten', nn.Flatten())
        #     inftrs = model(torch.zeros(5,3,224,224).to(device)).shape[1]
        #     classifier = nn.Linear(in_features=inftrs, out_features=10, bias=True)
        #     model.add_module('classifier', classifier)
            
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
    for sal_reg, rand_noise, balanced in [[True, True, False], [False, False, True], [True, True, True], [False, False, False]]:
        for mtype in ['timm_deit_small_patch16_224', 'torch_resnet50']:
            # for dset_name in ['rival10', 'rival20', 'hard_imagenet']:
            for dset_name in ['hard_imagenet']:
                ft = FineTuner(mtype=mtype, dset_name=dset_name,
                        sal_reg=sal_reg, rand_noise=rand_noise, balanced_subset=balanced)
                ft.finetune()
