# GAID: Generative Adversarial Irregularity Detection

Tensorflow implementation of [GAID](https://link.springer.com/chapter/10.1007/978-3-030-32281-6_10)

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

## Datasts

**- [Mias](http://peipa.essex.ac.uk/info/mias.html):** This dataset contains of 322 mammography images in MLO view with 1024*1024 resolution. The data is categorized into 3 classes: Benign, Malignant, and normal. The ground-trust of abnormal (Benign and Malignant tumor) regions are indicated by center and diameter of those regions

**- [INbreast](https://www.ncbi.nlm.nih.gov/pubmed/22078258):** This dataset contains 410 mammography images in mediolateraloblique
(MLO) and cranial-caudal (CC) views with a 30004000 resolution.We consider all the mass cases in this dataset as irregular versus the normal class present in the dataset.

**- [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#fa7d4f2e58a64fbaaab671105caa85f4):** This dataset contains 2,620 scanned film mammography studies from both CC and MLO views. The labels in this dataset also include benign, malignant, and normal with verified pathology information.

## Training

To train the model on MIAS dataset with preparing patches and create train and test datasets, run the following:
```
python main.py --dataset=mias --input_height=64 --output_height=64 --patch_size=64 --preparing_data --train
```

## Testing
To test the model on the MIAS test dataset prepared in train step, run the following:
```
python main.py --dataset=mias --input_height=64 --output_height=64 --test
```

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

