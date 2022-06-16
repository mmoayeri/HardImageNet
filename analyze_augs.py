from augmentations import *
from utils import *
from datasets import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
'''
Here, we analyze the effect of augmentation on object retention.
We track the following metrics:
- percent of image that is object
- percent of object that is retained after augmentation (compared to original image)

We can also visualize the average object masks to give a sense to where the object is
'''

def pretty_dset_name(dset_name):
    return dset_name.replace('_', ' ').title().replace('net','Net').replace('Rival', 'RIVAL')

def heatmaps_avg_obj_position(aug, aug_name, dset_name='hard_imagenet'):
    ''' this can only work for augs that include a resizing to standard size '''
    dset = HardImageNet(aug=aug) if dset_name=='hard_imagenet' else RIVAL10(aug=aug)
    masks_by_class = dict()
    trials = 10 if aug == random_resized_crop else 1
    for t in range(trials):
        for i in tqdm(range(len(dset))):
            _, mask, label = dset[i]
            if label not in masks_by_class:
                masks_by_class[label] = []
            masks_by_class[label].append(mask)

    ncol = 3 if dset_name=='hard_imagenet' else 4
    nrow=5
    f, axs = plt.subplots(5,ncol, figsize=(ncol*3, 15))

    _ = [axi.set_axis_off() for axi in axs.ravel()]
    imagenet_classnames = load_meta_files('imagenet_classnames')
    for i, c in enumerate(masks_by_class):
        avg_mask = torch.stack(masks_by_class[c]).mean(0).numpy()[0]#.swapaxes(0,1).swapaxes(1,2)
        im = axs[i % nrow, i // nrow].imshow(avg_mask, cmap='jet', vmin=0, vmax=1)
        cls_name = ' '.join(imagenet_classnames[c].title().split(' ')[-2:])
        axs[i % nrow, i // nrow].set_title(cls_name, fontsize=15)
    st = f.suptitle(f'{pretty_dset_name(dset_name)}', y=0.1, fontsize=30)
    f.subplots_adjust(wspace=0.15, hspace=0.15)#, right=0.8)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # if dset_name == 'hard_imagenet':
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)
    f.savefig(f'plots/avg_positions2_{aug_name}_{dset_name}.jpg', dpi=300, bbox_inches='tight', pad_inches=0.3, extra_artists=[st])


def percent_of_image_that_is_obj(aug, aug_name):
    dset = HardImageNet(aug=aug)
    percents_by_class = dict()
    for i in tqdm(range(len(dset))):
        _, mask, label = dset[i]
        if label not in percents_by_class:
            percents_by_class[label] = []
        num_pixels = 1
        for s in list(mask.shape):
            num_pixels *= s 
        percents_by_class[label].append(mask.sum() / num_pixels)

    f, ax = plt.subplots(1,1, figsize=(18,3))
    imagenet_classnames = load_meta_files('imagenet_classnames')
    colors = [cmap.get_cmap('jet')(x/15) for x in range(15)]
    xticks, xticklabels = [], []

    all_stats = [percents_by_class[c] for c in percents_by_class]
    positions = [i+1 for i in range(15)]
    parts = ax.violinplot(all_stats, positions, widths=0.4, showmeans=True, showextrema=True)
    for p,c in zip(parts['bodies'],colors):
        p.set_facecolor(c)
        p.set_edgecolor(c)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([imagenet_classnames[c].title() for c in percents_by_class], rotation=45)
    ax.set_ylabel('Percent of Image that is Object')
    f.savefig(f'plots/percent_obj_{aug_name}.jpg', bbox_inches='tight', pad_inches=0.05)

def compute_and_save_percent_img_that_is_obj():
    results = load_cached_results('percent_img_that_is_obj')
    for dset_name in ['hard_imagenet', 'rival10']:
        dset = get_dset(dset_name, ft=False, bs=1)
        if dset_name not in results:
            results[dset_name] = dict()
        for aug, aug_name in [(to_tens, 'No Augmentation'), (standard_resize_center_crop, 'Resize and Center Crop')]:
            dset.aug = aug
            percents = []
            trials = 10 if aug == random_resized_crop else 1
            for trial in range(trials):
                for i in tqdm(range(len(dset))):
                    _, mask, label = dset[i]
                    num_pixels = 1
                    for s in list(mask.shape):
                        num_pixels *= s 
                    percents.append(mask.sum().item() / num_pixels)
            results[dset_name][aug_name] = percents
            cache_results('percent_img_that_is_obj', results)                

def compute_and_save_percent_obj_retained():
    results = load_cached_results('percent_img_that_is_obj')
    resize = transforms.Resize(256)
    center_crop = transforms.CenterCrop(224)
    for dset_name in ['hard_imagenet', 'rival10']:
        dset_no_aug = get_dset(dset_name, ft=False)
        dset_no_aug.aug = to_tens
        percents = []
        for i in tqdm(range(len(dset_no_aug))):
            _, og_mask, _ = dset_no_aug[i]
            resized_mask = resize(og_mask)
            resized_and_cropped_mask = center_crop(resized_mask)

            percent_retained = (resized_and_cropped_mask.sum() / resized_mask.sum()).item()
            percents.append(percent_retained)
        results[dset_name] = percents
        print(f'Avg: {np.nanmean(percents)}, Dset: {dset_name}')
        cache_results('percent_obj_retained', results)   

def percent_of_image_that_is_obj_together(dset_name='hard_imagenet'):
    augs = [to_tens, standard_resize_center_crop, random_resized_crop]
    aug_names = ['None', 'Resized Crop', 'Random Resized Crop']

    all_stats = []
    dset = HardImageNet() if dset_name == 'hard_imagenet' else RIVAL10()
    for aug in augs:
        dset.aug = aug
        percents_by_class = dict()
        trials = 10 if aug == random_resized_crop else 1
        for trial in range(trials):
            for i in tqdm(range(len(dset))):
                _, mask, label = dset[i]
                if label not in percents_by_class:
                    percents_by_class[label] = []
                num_pixels = 1
                for s in list(mask.shape):
                    num_pixels *= s 
                percents_by_class[label].append(mask.sum() / num_pixels)
            stats = [percents_by_class[c] for c in percents_by_class]
        all_stats.append(stats)


    f, ax = plt.subplots(1,1, figsize=(dset.num_classes+1.5,3))
    imagenet_classnames = load_meta_files('imagenet_classnames')
    colors = [cmap.get_cmap('jet')(x/15) for x in range(dset.num_classes)]
    xticks, xticklabels = [], []

    positions = [2*i+1 for i in range(dset.num_classes)]
    pos2 = [p-0.5 for p in positions]
    pos3 = [p+0.5 for p in positions]

    for stats, c, pos in zip(all_stats, ['red','green', 'blue'], [pos2, positions, pos3]):
        parts = ax.violinplot(stats, pos, widths=0.35, showmeans=True, showextrema=True)
        for p in parts['bodies']:
            p.set_facecolor(c)
            p.set_edgecolor(c)
        

    # parts = ax.violinplot(all_stats, positions, widths=0.4, showmeans=True, showextrema=True)
    # for p,c in zip(parts['bodies'],colors):
    #     p.set_facecolor(c)
    #     p.set_edgecolor(c)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([imagenet_classnames[c].title() for c in percents_by_class], rotation=45)
    ax.set_ylabel('Percent of Image that is Object')

    f.savefig(f'plots/percent_obj_{dset_name}.jpg', bbox_inches='tight', pad_inches=0.05)


if __name__=='__main__':
    for dset_name in ['RIVAL10', 'hard_imagenet']:
        heatmaps_avg_obj_position(aug=standard_resize_center_crop, aug_name='standard', dset_name=dset_name)
    #     heatmaps_avg_obj_position(aug=random_resized_crop, aug_name='Random Resized Crop', dset_name=dset_name)
    #     percent_of_image_that_is_obj_together(dset_name=dset_name)

    # percent_of_image_that_is_obj(aug=standard_resize_center_crop, aug_name='Standard')
    # percent_of_image_that_is_obj(aug=random_resized_crop, aug_name='Random Resized Crop')
    # percent_of_image_that_is_obj(aug=None, aug_name='None')

    # compute_and_save_percent_img_that_is_obj()
    # compute_and_save_percent_obj_retained()