import numpy as np
import torch
from torchvision import transforms, models
import timm
from utils import *
from datasets import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from finetuner import FineTuner

def l2_normalize(X):
    flat_X = X.flatten(1)
    denom = flat_X.pow(2).sum(keepdim=True, dim=1).sqrt()
    out = flat_X / denom
    return out.reshape(X.shape)

def compute_rfs(noisy_fg_acc, noisy_bg_acc):
    avg = 0.5*(noisy_fg_acc + noisy_bg_acc)
    return 0 if (avg == 1 or avg == 0) else (noisy_bg_acc - noisy_fg_acc) / (2*min(avg, 1-avg))

def noisy_fg_bg_accs(model, dset_name='hard_imagenet', noise_sigma=0.25, apply_norm=True, l2=False, ft=False, trials=3):
    dset = HardImageNet(ft=ft) if dset_name == 'hard_imagenet' else RIVAL10(ft=ft, twenty=('20' in dset_name))
    loader = torch.utils.data.DataLoader(dset, batch_size=36, shuffle=True, num_workers=8, pin_memory=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cnt_by_class, noisy_fg_cc_by_class, noisy_bg_cc_by_class = dict(), dict(), dict()

    if noise_sigma == 0:# or noise_type == 'ablation':
        num_trials = 1

    ctr= 0
    for imgs, masks, labels in tqdm(loader):
        if noise_sigma > 0:
            masks = masks.cuda()
        imgs = imgs.cuda()
        labels = labels.cuda()

        for trial in range(trials):

            if noise_sigma > 0:
                noise = torch.randn_like(imgs, device=imgs.device)
                if l2:
                    noisy_fg, noisy_bg = [torch.clamp(imgs + noise_sigma*l2_normalize(x * noise), 0, 1) for x in [masks, 1-masks]]
                else: # linf noise
                    noisy_fg, noisy_bg = [torch.clamp(imgs + (x * noise * noise_sigma), 0, 1) for x in [masks, 1-masks]]

                if apply_norm:
                    noisy_fg, noisy_bg = [normalize(x) for x in [noisy_fg, noisy_bg]]
                noisy_fg_preds, noisy_bg_preds = [model(x).argmax(1) for x in [noisy_fg, noisy_bg]]
            else:
                if apply_norm:
                    imgs = normalize(imgs)
                preds = model(imgs).argmax(1)

            for y in np.unique(labels.cpu().numpy()):
                if y not in noisy_fg_cc_by_class:
                    noisy_fg_cc_by_class[y], noisy_bg_cc_by_class[y], cnt_by_class[y] = 0, 0, 0

                if noise_sigma > 0:
                    noisy_fg_cc_by_class[y] += (noisy_fg_preds[labels == y] == y).sum().item()
                    noisy_bg_cc_by_class[y] += (noisy_bg_preds[labels == y] == y).sum().item()
                else:
                    noisy_fg_cc_by_class[y] += (preds[labels == y] == y).sum().item()
                cnt_by_class[y] += (labels == y).sum().item()

        # if ctr > 5:
        #     break
        # ctr += 1
    
    # cnt = sum(cnt_by_class.values())
    # noisy_fg_acc, noisy_bg_acc = [sum(d.values())/cnt for d in [noisy_fg_cc_by_class, noisy_bg_cc_by_class]]
    noisy_fg_acc_by_class = dict({c:noisy_fg_cc_by_class[c]/cnt_by_class[c] for c in cnt_by_class})
    noisy_bg_acc_by_class = dict({c:noisy_bg_cc_by_class[c]/cnt_by_class[c] for c in cnt_by_class})
    noisy_fg_acc, noisy_bg_acc = [np.average(list(x.values())) for x in [noisy_fg_acc_by_class, noisy_bg_acc_by_class]]

    # core_acc, spur_acc = [100. * x / total_cnt for x in [total_core_cc, total_spur_cc]]
    rfs = compute_rfs(noisy_fg_acc, noisy_bg_acc)
    return rfs, noisy_fg_acc, noisy_bg_acc, noisy_fg_acc_by_class, noisy_bg_acc_by_class


# from torchvision import models
# net = models.resnet50(pretrained=True).eval().cuda()

# import timm
# net= timm.create_model('deit_small_patch16_224', pretrained=True).eval().cuda()

# for l2 in [True, False]:
#     print(f"{'' if l2 else 'NO'} L2 NORMALIZATION")
#     for dset_name in ['hard_imagenet', 'rival10']:
#         rfs, a,b,_,_ = noisy_fg_bg_accs(net, dset_name=dset_name, l2=l2, noise_sigma=100 if l2 else 0.25)
#         print(f'Dset: {dset_name:<30}, Noisy fg acc: {a:.3f}, Noisy bg acc: {b:.3f}, RFS: {rfs:.3f}')
# a,b,_,_ = rfs(net, dset=RIVAL10())
# print(f'Noisy fg acc: {a:.3f}, Noisy bg acc: {b:.3f}')

model_list=['torch_resnet50', 'timm_deit_small_patch16_224']
l2_sigmas = [30,60,90,120,150,180]
linf_sigmas = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]

# def eval_pretrained_models(model_list=['torch_resnet50', 'robust_resnet50_l2_eps3', 'timm_deit_small_patch16_224', 'simclr']):
def eval_pretrained_models():
    for ft in [False, True]:
        results_key = 'rfs{}'.format('_finetuned' if ft else '')
        results = load_cached_results(results_key)
        for mkey in model_list:
            # load model
            if not ft:
                if mkey == 'torch_resnet50':
                    model = models.resnet50(pretrained=True).eval().cuda()
                elif mkey == 'timm_deit_small_patch16_224':
                    model = timm.create_model(mkey[len('timm_'):], pretrained=True).eval().cuda()

            # eval model
            apply_norm = True
            if mkey not in results:
                results[mkey] = dict()
            for l2, sigma_list in zip([True, False], [l2_sigmas, linf_sigmas]):
                norm = 'l2' if l2 else 'linf'
                if norm not in results[mkey]:
                    results[mkey][norm] = dict()

                for sigma in sigma_list:
                    if sigma not in results[mkey][norm]:
                        results[mkey][norm][sigma] = dict()
                    for dset_name in ['hard_imagenet', 'rival10', 'rival20']:
                        if dset_name not in results[mkey][norm][sigma]:
                            if ft:
                                finetuner = FineTuner(mtype=mkey, dset_name=dset_name)
                                finetuner.restore_model()
                                model = finetuner.model
                            elif dset_name == 'rival20':
                                continue
                            rfs, noisy_fg_acc, noisy_bg_acc, noisy_fg_acc_by_class, noisy_bg_acc_by_class = \
                                noisy_fg_bg_accs(model, dset_name, sigma, apply_norm, l2, ft)
                            results[mkey][norm][sigma][dset_name] = dict({
                                'noisy_fg_acc': noisy_fg_acc, 'noisy_bg_acc': noisy_bg_acc, 'rfs': rfs,
                                'noisy_fg_acc_by_class': noisy_fg_acc_by_class, 'noisy_bg_acc_by_class': noisy_bg_acc_by_class
                            })
                            cache_results(results_key, results)
                        print('Model: {:<40}, Norm: {:<5}, Dset: {:<15}, Sigma: {:<5}'.format(mkey, norm, dset_name, sigma) + 
                            '....Noisy FG Acc: {:.3f}....Noisy BG Acc: {:.3f}....RFS:{:.3f}'.format(
                                *[results[mkey][norm][sigma][dset_name][x] for x in ['noisy_fg_acc', 'noisy_bg_acc', 'rfs']]
                            ))

def plot_rfs(ft=False):
    results = load_cached_results('rfs{}'.format('_finetuned' if ft else ''))
    f, axs = plt.subplots(2, len(model_list), figsize=(5*len(model_list), 7))
    for i, mkey in enumerate(model_list):
        for j, (norm, sigmas) in enumerate(zip(['l2', 'linf'], [l2_sigmas, linf_sigmas])):
            rival10_rfs= [results[mkey][norm][sigma]['rival10']['rfs'] for sigma in sigmas]
            hard_imagenet_rfs= [results[mkey][norm][sigma]['hard_imagenet']['rfs'] for sigma in sigmas]

            axs[i,j].plot(sigmas, rival10_rfs, '-*', c='deepskyblue', label='RIVAL10')
            axs[i,j].plot(sigmas, hard_imagenet_rfs, '-o', c='coral', label='Hard ImageNet')
            axs[i,j].legend()
            axs[i,j].set_xlabel('$\ell_2$ Norm of Added Noise' if norm == 'l2' else 'Std. Dev. $\sigma$ of $\ell_\infty$ Gaussian Noise')
            axs[i,j].set_ylabel('Relative Foreground Sensitivity ($RFS$)')
            axs[i,j].set_title(mkey)
    f.tight_layout(); f.savefig('plots/rfs{}.jpg'.format('_finetuned' if ft else ''), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    eval_pretrained_models()
    # plot_rfs(ft=True)