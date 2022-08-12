# HardImageNet

This is the repository for the Hard ImageNet dataset, associated with the NeurIPS 2022 Datasets and Benchmarks submission, <em>``Hard ImageNet: Segmentations for Objects with Strong Spurious Cues''</em>. Read more on our [website](mmoayeri.github.io/HardImageNet).

## Download the dataset

You can download the dataset directly from box by following this [link](https://umd.app.box.com/s/ca7qlcfsqlfqul9rzgtuqhb2c6pm62qd). 

Alternatively, the dataset can be downloaded from the command line as follows:
    
    curl -L 'https://app.box.com/index.php?rm=box_download_shared_file&shared_name=ca7qlcfsqlfqul9rzgtuqhb2c6pm62qd&file_id=f_972129165893' -o hardImageNet.zip
    unzip hardImageNet.zip
    
The dataset should contain directories for the train and validation splits (named `train' and 'val' respectively), as well as a pickle file containing the class-wise rankings of the strength of spurious cues present for each sample in the training set. 

## Setting up the data

Be sure to update lines 10 and 11 in datasets/hard_imagenet.py so that they point to the ImageNet path and the HardImageNet path in your files respectively. Then, you can simply instantiate a HardImageNet dataset object using the HardImageNet() constructor, after having included the line 'from datasets import *'. 

Alternatively, you can use the function get_dset in utils.py. 

## Evaluate Models

We provide code for three evaluations using Hard ImageNet. To assess the degree to which a model relies on spurious cues, you can:
1. Ablate the object and assess the drop in accuracy (low drop indicates high spurious feature dependence). Code for this evaluation is in ablations.py
2. Noise the object and the background and compare performance using <em>Relative Foreground Sensitivity</em> metric. Code for this evaluation is in rfs.py
3. Compute the alignment of saliency to the object region via intersection over union of GradCAM and object segmentation. Code for this evaluation is in saliency_analysis.py

## Citation

If the dataset or code is of use to you, please consider citing:

    @misc{moayeri2022hard,
        title     = {Hard ImageNet: Segmentations for Objects with Strong Spurious Cues},
        author    = {Moayeri, Mazda and Singla, Sahil and Feizi, Soheil},
        booktitle = {openreview},
        month     = {June},
        year      = {2022},
    }
