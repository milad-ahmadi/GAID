# GAID: Generative Adversarial Irregularity Detection

Tensorflow implementation of [GAID](https://openreview.net/forum?id=Ske2oyiye4)

This model detect irregular part in mammography images, after training GAID with normal dataset.

## Model Description

Recognizing irregular tissues in mammography images can be defined as discovering regions that does not comply with normal (healthy) tissues present in the training set. To this end, we propose a method based on adversarial training composed of two important modules. The first module, denoted by R (Reconstructor), discovers the distribution of the healthy tissues, by learning to reconstruct them. The second module, M (Representation matching), learn to detect if its input is healthy or irregular.

<p align="center">
  <img src="https://github.com/milad-ahmadi/GAID/blob/master/images/R%2BM.PNG">
</p>


## Prerequisites (my environments)
- Windows, Linux or macOS
- Python 3.6
- Tensorflow 
- SciPy
- Pandas 
- matplotlib
- scikit-learn
- seaborn
- CPU or NVIDIA GPU + CUDA CuDNN

## Detests

**- [Mias](http://peipa.essex.ac.uk/info/mias.html):** This dataset contains of 322 mammography images in MLO view with 1024*1024 resolution. The data is categorized into 3 classes: (1)Benign, (2)Malignant, and (3) normal. The ground-trust of abnormal (Benign and Malignant tumor) regions are indicated by center and diameter of those regions

**- [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#fa7d4f2e58a64fbaaab671105caa85f4):** This dataset contains 2,620 scanned film mammography studies from both cranial-caudal (CC) and mediolateral-oblique (MLO) views. The labels in this dataset also include benign, malignant, and normal with veried pathology information.

## Training

## Testing

## Results
Examples of patches (denoted by X) and their reconstructed versions using AnoGAN, GANomaly ,and GAID.
<p align="center">
  <img src="https://github.com/milad-ahmadi/GAID/blob/master/images/reconstructed results.PNG">
</p>

Testing results of the proposed irregularity detector on the CBIS-DDSM dataset, trained on MIAS dataset.
<p align="center">
  <img src="https://github.com/milad-ahmadi/GAID/blob/master/images/heat-map results.PNG">
</p>


## To Do

## Acknowledgement

## Reference

