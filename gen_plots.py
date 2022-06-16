from utils import *
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

mkeys = ['torch_resnet50', 'timm_deit_small_patch16_224']
mnames = ['ResNet50', 'DeiT (Small)']
short_mnames = ['RN50', 'DeiT']
markers = ['^', 's'] # for models
line_styles = ['--', '-.'] # for ft vs not ft
colors = dict({
    'hard_imagenet':'firebrick', 
    'rival10':'aqua', 
    'rival20':'deepskyblue'
}) # for datasets
no_ft_dsets = ['hard_imagenet', 'rival10']
ft_dsets = ['hard_imagenet', 'rival10', 'rival20']


def pretty_dset_name(dset_name):
    return dset_name.replace('_', ' ').title().replace('net','Net').replace('Rival', 'RIVAL')

#### Ablations
def ablations_bar_plots():
    '''
    Plan is to have 2 rows, one per model type, and 2 cols (for ft or not)
    In each subfig, we have xticks for each ablation type, and bars for dsets (labelled in legend)
    '''
    plt.style.use('ggplot')
    results = load_cached_results('ablations')

    ab_names = ['none', 'replace_with_gray', 'replace_bbox_with_gray', 'tile']
    pretty_ab_names = ['None', 'Object\nGrayed', 'BBox\nGrayed', 'Tiled']

    # non-ft models only have two datasets, so we make it a bit smaller
    m=1.3
    f, axs = plt.subplots(2,1, figsize=(4,6))
    for ax, mkey, mname in zip(axs, mkeys, mnames):
        for i, ab_name in enumerate(ab_names):
            ax.bar(m*i+1-0.25, results[ab_name][False][mkey]['hard_imagenet'], label='Hard ImageNet', width=0.5, color=colors[0])
            ax.bar(m*i+1+0.25, results[ab_name][False][mkey]['rival10'], label='RIVAL20', width=0.5, color=colors[2])

        ax.set_xticks(m*np.arange(1,1+len(ab_names))-0.3)
        ax.set_xticklabels(pretty_ab_names)
        ax.legend(handles=[Patch(color=c, label=l) for l,c in [('Hard ImageNet',colors[0]), ('RIVAL20', colors[2])]])
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Ablation')
        ax.set_title(mname, fontsize=13)
    # axs[1].set_title('DeiT (Small)', fontsize=13)
    f.tight_layout(); f.savefig('plots/ablation_no_ft.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    f, axs = plt.subplots(2,1, figsize=(4,6))
    for ax, mkey, short_mname in zip(axs, mkeys, short_mnames):
        for i, ab_name in enumerate(ab_names):
            ax.bar(m*i+1-0.3, results[ab_name][True][mkey]['hard_imagenet'], label='Hard ImageNet', width=0.3, color=colors[0])
            ax.bar(m*i+1, results[ab_name][True][mkey]['rival10'], label='RIVAL10', width=0.3, color=colors[1])
            ax.bar(m*i+1+0.3, results[ab_name][True][mkey]['rival20'], label='RIVAL20', width=0.3, color=colors[2])

        ax.set_xticks(m*np.arange(1,1+len(ab_names))-0.3)
        ax.set_xticklabels(pretty_ab_names)
        ax.set_xlabel('Ablation')
        ax.set_ylabel('Accuracy')
        ax.legend(handles=[Patch(color=c, label=l) for l,c in [('Hard ImageNet',colors[0]), ('RIVAL10', colors[1]), ('RIVAL20', colors[2])]], loc='lower left')
        ax.set_title(f'Finetuned {short_mname}', fontsize=13)
    # axs[1].set_title('Finetuned DeiT', fontsize=13)
    f.tight_layout(); f.savefig('plots/ablation_ft.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)

def ablation_examples():
    f,axs = plt.subplots(4,1,figsize=(2,6))
    _ = [axi.set_axis_off() for axi in axs.ravel()]
    og, grayed_obj, grayed_bbox, tiled = [mpimg.imread(f'./plots/ablation_egs/{x}.png') for x in ['og', 'obj_grayed', 'bbox_grayed', 'tiled']]
    
    axs[0].imshow(og); axs[0].set_title('Original', fontsize=10)
    axs[1].imshow(grayed_obj); axs[1].set_title('Object Grayed', fontsize=10)
    axs[2].imshow(grayed_bbox); axs[2].set_title('BBox Grayed', fontsize=10)
    axs[3].imshow(tiled); axs[3].set_title('BBox Tiled', fontsize=10)
    f.tight_layout(); f.savefig('plots/ablation_fig_egs.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)


### RFS Plots
def plot_rfs():
    plt.style.use('ggplot')
    l2_sigmas = [30,60,90,120,150,180]
    linf_sigmas = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]

    for ft, dsets in zip([True, False], [ft_dsets, no_ft_dsets]):
        results = load_cached_results('rfs{}'.format('_finetuned' if ft else ''))
        f, axs = plt.subplots(1,2, figsize=(8,5))
        dsets.reverse()
        for j, (norm, ax, sigmas) in enumerate(zip(['l2', 'linf'], axs, [l2_sigmas, linf_sigmas])):
            for mkey,short_mname, marker in zip(mkeys, short_mnames, markers):
                handles = []
                for dset in dsets:
                    vals = [results[mkey][norm][sigma][dset]['rfs'] for sigma in sigmas]
                    dset = 'rival20' if dset == 'rival10' and not ft else dset
                    ax.plot(sigmas, vals, '-'+marker, c=colors[dset])#, label=f'{short_mname}, {pretty_dset_name(dset)}')
                    # legend_elements.append(Line2D([0],[0],marker=marker, label=f'{short_mname}, {pretty_dset_name(dset)}', c=colors[dset]))
                    handles.append(Patch(color=colors[dset], label=pretty_dset_name(dset)))
            # ax.legend(handles=[Patch(color=colors[d], label=pretty_dset_name(d)) for d in dsets], fontsize=13)
            ax.legend(handles=handles, fontsize=13)
            ax.set_xlabel('$\ell_2$ Norm of Added Noise' if norm == 'l2' else 'Std. Dev. $\sigma$ of $\ell_\infty$ Gaussian Noise')
            ax.set_ylabel('Relative Foreground Sensitivity ($RFS$)')
        f.suptitle('{} DeiT (squares) and ResNet (triangles)'.format('Finetuned' if ft else 'Off the Shelf'), fontsize=16, y=0.98)
        f.tight_layout();f.savefig('plots/rfs_merged{}.jpg'.format('_ft' if ft else ''), bbox_inches='tight', pad_inches=0.1, dpi=300)


### Saliency Alignment violin
def saliency_violins():
    # one long fig, w non-ft models followed by ft models, one violin per dset per (model, ft) combo
    results = load_cached_results('ious_and_delta_densities')
    print(results.keys())
    all_ious, positions, cs = [], [], []
    i = 0
    m = 2
    for ft in [False, True]:
        for mkey in mkeys:
            i += 1
            all_ious.append(list(results[ft][mkey]['hard_imagenet'][0].values()))
            cs.append(colors['hard_imagenet'])
            if ft:
                all_ious.append(list(results[ft][mkey]['rival10'][0].values()))
                cs.append(colors['rival10'])
                positions.append(m*i-0.5)
                positions.append(m*i)
                positions.append(m*i+0.5)

            else:    
                positions.append(m*i-0.3)
                positions.append(m*i+0.3)

            all_ious.append(list(results[ft][mkey]['rival10' if not ft else 'rival20'][0].values())) # key is 'rival10' but we're gonna call it rival20 if not ft
            cs.append(colors['rival20'])

    plt.style.use('ggplot')
    f, ax = plt.subplots(1,1, figsize=(10, 4))
    vparts = ax.violinplot(all_ious, positions, widths=0.45, showextrema=True, showmedians=True)
    for i,part in enumerate(vparts['bodies']):
        part.set_color(cs[i])
    
    handles = [Patch(color=colors[d], label=pretty_dset_name(d)) for d in ['hard_imagenet', 'rival10', 'rival20']]
    ax.legend(handles=handles)
    
    ax.set_xticks(m*np.arange(1,5))
    ax.set_xticklabels(['Off the Shelf ResNet50', 'Off the Shelf DeiT (Small)', 'Finetuned RN50', 'Finetuned DeiT'])
    ax.set_ylabel('Saliency Alignment (IoU)')
    f.tight_layout(); f.savefig('plots/sal_alignment.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)

def interesting_saliency_egs():
    root = '/scratch1/mmoayeri/hard_imagenet/bad_gradcams/delta_density/hard_imagenet/'
    paths = [('class_6/torch_resnet50_ft_10.jpg', 'Balance Beam'),
             ('class_0/torch_resnet50_ft_6.jpg', 'Dog Sled'),
             ('class_9/torch_resnet50_ft_3.jpg', 'Hockey Puck'),
             ('class_10/timm_deit_small_patch16_224_ft_85.jpg', 'Mini Skirt'),
             ('class_11/timm_deit_small_patch16_224_ft_2.jpg', 'Keyboard Spacebar')]
    f, axs = plt.subplots(1, len(paths), figsize=(4*len(paths), 4))
    _ = [axi.set_axis_off() for axi in axs.ravel()]
    for ax, (p, cls_name) in zip(axs, paths):
        img = mpimg.imread(root+p)
        ax.imshow(img)
        ax.set_title(cls_name, fontsize=20)
    f.tight_layout(); f.savefig('plots/cool_saliencies.jpg', dpi=300, bbox_inches='tight', pad_inches=0.5)


### On Object position/stats of segmentation masks
def percent_img_that_is_obj_violins():
    results = load_cached_results('percent_img_that_is_obj')
    stats, cs = [], []
    for i, aug_name in enumerate(['No Augmentation', 'Resize and Center Crop']):
        for dset_name in ['hard_imagenet', 'rival10']:
            stats.append(results[dset_name][aug_name])
            cs.append(colors[dset_name])

    plt.style.use('ggplot')
    f, ax = plt.subplots(1,1, figsize=(6,5))
    vparts = ax.violinplot(stats, [1.6,2.4, 3.6, 4.4], widths=0.4, showextrema=True, showmedians=True)
    for i,part in enumerate(vparts['bodies']):
        part.set_color(cs[i])
    
    handles = [Patch(color=colors[d], label=pretty_dset_name(d)) for d in ['hard_imagenet', 'rival20']]
    ax.legend(handles=handles, fontsize=12)

    ax.set_xticks([2,4])
    ax.set_xticklabels(['No Augmentation', 'Resize and Center Crop'], fontsize=14)
    ax.set_ylabel('Percent of Image that is Object', fontsize=16)
    f.tight_layout(); f.savefig('plots/percent_img_that_is_obj.jpg', dpi=300, bbox_inches='tight', pad_inches=0.5)

def percent_obj_retained_violins():
    results = load_cached_results('percent_obj_retained')
    stats, cs = [], []
    for dset_name in ['hard_imagenet', 'rival10']:
        percents = results[dset_name]
        percents = [0 if np.isnan(x) else x for x in percents]
        stats.append(percents)
        cs.append(colors[dset_name])

    plt.style.use('ggplot')
    f, ax = plt.subplots(1,1, figsize=(6,5))
    vparts = ax.violinplot(stats, [2,4], widths=1, showextrema=True, showmedians=True)
    for i,part in enumerate(vparts['bodies']):
        part.set_color(cs[i])

    ax.set_xticks([2,4])
    ax.set_xticklabels([pretty_dset_name(d) for d in ['hard_imagenet', 'rival20']], fontsize=18)
    ax.set_ylabel('Percent of Object that is Retained\nAfter Resize and Center Crop', fontsize=16)
    f.tight_layout(); f.savefig('plots/percent_obj_retained.jpg', dpi=300, bbox_inches='tight', pad_inches=0.5)

def copy_imgs_for_dataset_fig():
    ''' simply shows instance of each class with segmentation '''
    hard_imagenet_idx, imagenet_classnames = [load_meta_files(x) for x in ['hard_imagenet_idx', 'imagenet_classnames']]
    num_to_show = dict({imagenet_classnames[ind].replace(' ', '_'):5 for ind in hard_imagenet_idx})
    # some adjustments to remove imgs prominently showing faces
    num_to_show['gymnastics_horizontal_bar'] = 4
    num_to_show['keyboard_space_bar'] = 3
    num_to_show['miniskirt'] = 2
    num_to_show['patio'] = 1
    num_to_show['seat_belt'] = 2
    num_to_show['snorkel'] = 4
    num_to_show['sunglasses'] = 3
    num_to_show['swimming_cap'] = 1
    num_to_show['hockey_puck'] = 4

    import shutil
    root = 'data_collection/ground_truths/overlayed/{}/{}.png'
    dest = 'plots/example_segmentations/{}.png'

    from augmentations import standard_resize_center_crop
    f, axs = plt.subplots(3,5, figsize=(15,9))
    _ = [axi.set_axis_off() for axi in axs.ravel()]
    for ind, ax in zip(hard_imagenet_idx, axs.ravel()):
        cls_name = imagenet_classnames[ind].replace(' ', '_')
        shutil.copy(root.format(cls_name, num_to_show[cls_name]), dest.format(cls_name))

        # img =  mpimg.imread(root.format(cls_name, num_to_show[cls_name]))
        img = Image.open(root.format(cls_name, num_to_show[cls_name]))
        resized,_ = standard_resize_center_crop(img, img)
        img = resized.numpy().swapaxes(0,1).swapaxes(1,2)
        ax.imshow(img)
        ax.set_title(' '.join(cls_name.title().split('_')[-2:]), fontsize=20)

    f.subplots_adjust(wspace=0.05, hspace=0.05)
    f.tight_layout(); f.savefig('plots/egs.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)
    

if __name__=='__main__':
    # ablations_bar_plots()
    # ablation_examples()
    # plot_rfs()

    saliency_violins()
    # interesting_saliency_egs()

    # percent_img_that_is_obj_violins()
    # percent_obj_retained_violins()

    # copy_imgs_for_dataset_fig()