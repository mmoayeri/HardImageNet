import torch
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from tqdm import tqdm
from datasets import *
import pickle
import matplotlib.cm as cmap
from utils import *
import timm
from finetuner import FineTuner
from torchvision import models, transforms
import os
#### UTILS

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def clip_vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[1 :  , :].reshape(tensor.size(1),
        height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2).float()
    return result

def swin_reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_cam_obj(model, target_layer, mtype, camtype='gradcam'):
    if 'clip_ViT' in mtype:
        if '32' in mtype:
            reshape_transform = lambda x : clip_vit_reshape_transform(x, height=7, width=7)
        else:
            reshape_transform = lambda x : clip_vit_reshape_transform(x, height=14, width=14)

    elif 'clip_RN' in mtype:
        reshape_transform = lambda x : x.float()
    elif 'swin' in mtype:
        reshape_transform = swin_reshape_transform
    elif 'deit' in mtype or 'vit' in mtype:
    # if use_vit_transform:
        reshape_transform = vit_reshape_transform
    else:
        reshape_transform = lambda x : x
    
    if camtype == 'gradcam':
        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True, reshape_transform=reshape_transform)
    elif camtype == 'scorecam':
        cam = ScoreCAM(model=model, target_layers=[target_layer], use_cuda=True, reshape_transform=reshape_transform)

    return cam

def binarize(m, thresh=0.5):
    m = np.array(m)
    m[np.isnan(m)] = 0
    m[m >= thresh] = 1
    m[m < thresh] = 0
    return m

### metrics

def delta_saliency_density(gcam, mask):
    density_fg = np.sum(gcam * mask) / np.sum(mask)
    density_bg = np.sum(gcam * (1-mask)) / np.sum(1-mask)
    return density_fg - density_bg

def intersection_over_union_and_dice(masks, eps=1e-10, thresh=0.5):
    '''
    takes list of two masks, both can be soft. Binarizes masks with threshold thresh.
    Computes IoU and Dice score, returning both 
    '''
    masks = [binarize(m, thresh) for m in masks]
    intersection = masks[0]
    for mask in masks[1:]:
        intersection = intersection * mask
    intersection = np.sum(intersection > 0)
    union = np.sum(np.sum(masks, axis=0) > 0)
    dice = (2*intersection + eps) / (union + intersection + eps) #if union != 0 else NaN
    iou = (intersection + eps) / (union + eps)
    return iou, dice

### obtaining scores

def compute_scores_one_batch(imgs, masks, target_labels, model, target_layer, cam=None):
    ''' For one batch, visualizes gradcams and returns IOU score '''

    # print([i.item() for i in target_labels])
    targets = [ClassifierOutputTarget(i.item()) for i in target_labels]
    grayscale_cam = cam(input_tensor=imgs, targets=targets, eigen_smooth=False)
    grayscale_cam[np.isnan(grayscale_cam)] = 0

    N = masks.shape[0]
    masks = masks[:,0]
    masks = masks.detach().cpu().numpy()

    # print((masks-grayscale_cam).shape)
    # dists = torch.norm(torch.tensor((masks-grayscale_cam)).flatten(1), p=2, dim=1)
    ious = [intersection_over_union_and_dice([masks[i], grayscale_cam[i]])[0] for i in range(N)]
    # fracs_inside = [np.sum(grayscale_cam[i] * masks[i]) / np.sum(grayscale_cam[i]) for i in range(N)]
    # mask_coverages = [compute_mask_coverage(masks[i], grayscale_cam[i]) for i in range(N)]
    delta_densities = [delta_saliency_density(grayscale_cam[i], masks[i]) for i in range(N)]
    # aps = [average_precision_score(grayscale_cam[i].flatten(), masks[i].flatten()) for i in range(N)]

    # scores = (dists, fracs_inside, mask_coverages, delta_densities, ious)
    return ious, delta_densities #scores 


##### 
def update_scores(scores_dict, scores):
    for i, scores_list in enumerate(scores):
        scores_dict[i].extend(scores_list)
    return scores_dict

def compute_salient_alignment_scores(dset, model, target_layer, mtype):
    '''
    We'll assume all models are resnets, who's gradcam layer is model.layer4[-1]

    *** IT IS IMPERATIVE THAT LOADER IS NOT SHUFFLED *** 
    so that we can compare across models later
    '''
    # SUPER IMPORTANT TO KEEP SHUFFLE=FALSE
    loader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    model = model.eval().cuda()
    cam = get_cam_obj(model, target_layer, mtype=mtype)
    all_ious, all_delta_densities = [], []
    dset_ids = []
    ctr = 0
    for imgs, masks, labels in tqdm(loader):
        
        idx_with_masks = (masks.flatten(1).sum(1) != 0)
        dset_ids.extend([ctr + i for i,x in enumerate(idx_with_masks) if x])
        ctr += imgs.shape[0]
        imgs, masks, labels = [x[idx_with_masks].cuda() for x in [imgs, masks, labels]]
        # imgs = normalize(imgs)
        # print(masks.shape, masks.flatten(1).sum(1))
        if masks.shape[0] == 0:
            continue
        
        ious, delta_densities = compute_scores_one_batch(imgs, masks, labels, model, target_layer, cam)
        all_ious.extend(ious)
        all_delta_densities.extend(delta_densities)
        # scores = compute_scores_one_batch(imgs, masks, labels, model, target_layer, cam)
        # scores_dict = update_scores(scores_dict, scores)
        # if ctr > 100:
        #     break
    ious_by_id = dict({i:iou for i, iou in zip(dset_ids, all_ious)})
    delta_densities_by_id = dict({i:iou for i, iou in zip(dset_ids, all_delta_densities)})
    return ious_by_id, delta_densities_by_id

# def print_average_scores(mkey, scores_dict):
#     ind_to_metric_name = scores_dict['ind_to_metric_name']
#     msg = 'Model: {:30}'.format(mkey)
#     for i in ind_to_metric_name:
#         msg += '\n{:10}: {:.2f}, std: {:.2f}'.format(ind_to_metric_name[i], 
#             np.average(scores_dict[i]), np.std(scores_dict[i]))
#     msg += '\n'
#     print(msg)

def eval():
    results = load_cached_results('ious_and_delta_densities')
    # delta_density_results = load_cached_results('delta_density')
    for ft in [False, True]:
        if ft not in results:
            results[ft] = dict()
        for mkey in ['torch_resnet50', 'timm_deit_small_patch16_224']:
            if mkey not in results[ft]:
                results[ft][mkey] = dict()
            for dset_name in ['hard_imagenet', 'rival10', 'rival20']:
                if dset_name == 'rival20' and not ft:
                    continue
                if dset_name not in results[ft][mkey]:
                    model, target_layer = get_model(mkey, dset_name, ft)
                    dset = get_dset(dset_name, ft)

                    results[ft][mkey][dset_name] = compute_salient_alignment_scores(dset, model, target_layer, mkey)
                    cache_results('ious_and_delta_densities', results)
                print('Finetuned: {:<8}, Model: {:<30}, Dset: {:<20}, Avg IoU: {:.3f}, Avg Delta density: {:.3f}'.format(
                    ft, mkey, dset_name, *[np.nanmean(list(results[ft][mkey][dset_name][x].values()))*100 for x in [0,1]]
                ))

#### PLOTS

def violin_plots(full_dict, arch, metric_ind):
    ''' full_dict takes forever to load, so just do it once ''' 
    l2_epsilons = [0, 0.25, 0.5, 1, 3, 5]
    linf_epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
    labels, dat = [], []
    for norm, epsilons in zip(['l2', 'linf'], [l2_epsilons, linf_epsilons]):
        for eps in epsilons:
            dat.append(full_dict[f'{arch}_{norm}_eps{eps}'][metric_ind])
            norm_nickname = norm.replace('l', 'l_').replace('inf', '\infty')
            labels.append("${}$, $\epsilon={}$".format(norm_nickname, eps))
    f, ax = plt.subplots(1,1)
    vparts = ax.violinplot(dat, np.arange(1,len(dat)+1), widths=0.75,
                     showextrema=True, showmedians=True)
    facecolors = [cmap.jet(x/len(l2_epsilons)) for x in range(len(l2_epsilons))]
    facecolors = facecolors + facecolors[1:] # so that linf and l2 violins have same color
    for i,part in enumerate(vparts['bodies']):
        part.set_color(facecolors[i])
    ax.set_xticks(np.arange(1,len(labels)+1))
    ax.set_xticklabels(labels, rotation='vertical')
    metric = full_dict['resnet18_l2_eps0']['ind_to_metric_name'][metric_ind]
    ax.set_ylabel(metric.title())
    ax.set_title(arch.title())
    f.tight_layout()
    f.savefig(f'./plots/sal_alignment/{arch}/2{metric}.jpg', dpi=150)

def view_bad_examples(dset_name='hard_imagenet', mkey='timm_deit_small_patch16_224', ft=True, num_to_save=10, metric='iou'):
    # all_results = load_cached_results('ious_w_norm')
    all_results = load_cached_results('ious_and_delta_densities')
    d = all_results[ft][mkey][dset_name][0 if metric =='iou' else 1]
    # we'll need to get the original instances that had bad gradcam iou
    dset = get_dset(dset_name, ft)
    # we'll need to regenerate the gradcams
    model, target_layer = get_model(mkey, dset_name, ft)
    cam = get_cam_obj(model, target_layer, mtype=mkey)

    idx = np.argsort(list(d.values()))
    bad_iou_dset_idx = np.array(list(d.keys()))[idx]
    to_pil = transforms.ToPILImage()


    cnt_per_class = dict()#[0]*dset.num_classes
    os.makedirs(f"bad_gradcams/{metric}/", exist_ok=True)
    os.makedirs(f"bad_gradcams/{metric}/{dset_name}", exist_ok=True)
    # for i in range(num_to_save):
    i=0
    while sum([int(c >= 3) for c in cnt_per_class]) < dset.num_classes and i < len(bad_iou_dset_idx):
        x,m,y = dset[bad_iou_dset_idx[i]]
        if y not in cnt_per_class:
            cnt_per_class[y] = 0
            os.makedirs(f"bad_gradcams/{metric}/{dset_name}/class_{y}/", exist_ok=True)
        if cnt_per_class[y] < 3:
            grayscale_cam = cam(input_tensor=normalize(x).unsqueeze(0), targets=[ClassifierOutputTarget(y)], eigen_smooth=False)
            grayscale_cam = cam(input_tensor=x.unsqueeze(0), targets=[ClassifierOutputTarget(y)], eigen_smooth=False)
            grayscale_cam[np.isnan(grayscale_cam)] = 0

            rgb_img = x.detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2)
            visualization = show_cam_on_image(rgb_img, grayscale_cam[0])
            img = Image.fromarray(visualization)
            r, g, b = img.split()
            img = Image.merge('RGB', (b, g, r))
            img.save(f"bad_gradcams/{metric}/{dset_name}/class_{y}/{mkey}{'_ft' if ft else ''}_{i}.jpg")
        # to_pil(x*m).save(f"bad_iou_gradcams/{mkey}{'_ft' if ft else ''}_{i}_og.jpg")
            cnt_per_class[y] += 1
        i += 1

if __name__ == '__main__':
    eval()
    # for metric in ['iou', 'delta_density']:
    #     view_bad_examples(metric=metric, mkey='torch_resnet50')
    #     view_bad_examples(metric=metric)