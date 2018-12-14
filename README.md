# GAID: Generative Adversarial Irregularity Detection

Tensorflow implementation of [GAID](https://openreview.net/forum?id=Ske2oyiye4)

This model detect irregular part in mammography images, after training GAID with normal dataset.

## Model Description

Recognizing irregular tissues in mammography images can be defined as discovering regions that does not comply with normal (healthy) tissues present in the training set. To this end, we propose a method based on adversarial training composed of two important modules. The first module, denoted by R (Reconstructor), discovers the distribution of the healthy tissues, by learning to reconstruct them. The second module, M (Representation matching), learn to detect if its input is healthy or irregular.

## Prerequisites (my environments)

## Detests

**Mias:**This dataset contains of 322 mammography images in MLO view with 1024*1024 resolution. The data is categorized into 3 classes: (1)Benign, (2)Malignant, and (3) normal. The ground-trust of abnormal (Benign and Malignant tumor) regions are indicated by center and diameter of those regions

## Training

### Train on Mias

## Testing

## Results

## To Do

## Acknowledgement

## Reference

