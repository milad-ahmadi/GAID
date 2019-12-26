# GAID: Generative Adversarial Irregularity Detection

Tensorflow implementation of [GAID](https://link.springer.com/chapter/10.1007/978-3-030-32281-6_10)

This model detect irregular part in mammography images, after training GAID with normal dataset.

## Model Description

Recognizing irregular tissues in mammography images can be defined as discovering regions that does not comply with normal (healthy) tissues present in the training set. To this end, we propose a method based on adversarial training composed of two important modules. The first module, denoted by R (Reconstructor), discovers the distribution of the healthy tissues, by learning to reconstruct them. The second module, M (Representation matching), learn to detect if its input is healthy or irregular.

<p align="center">
  <img src="https://github.com/milad-ahmadi/GAID/blob/master/images/R%2BM.PNG">
</p>


## Prerequisites (my environments)
- Python 3.6
- Tensorflow 
- SciPy = 1.0.0
- Pillow
- Pandas 
- matplotlib
- scikit-learn
- seaborn
- OpenCV
- [mritopng](https://github.com/danishm/mritopng) (Convert DICOM Files to PNG)
- CPU or NVIDIA GPU + CUDA CuDNN

## Datasets

**- [Mias](http://peipa.essex.ac.uk/info/mias.html):** This dataset contains of 322 mammography images in MLO view with 1024*1024 resolution. The data is categorized into 3 classes: Benign, Malignant, and normal. The ground-trust of abnormal (Benign and Malignant tumor) regions are indicated by center and diameter of those regions

**- [INbreast](https://www.ncbi.nlm.nih.gov/pubmed/22078258):** This dataset contains 410 mammography images in mediolateraloblique
(MLO) and cranial-caudal (CC) views with a 3000*4000 resolution.We consider all the mass cases in this dataset as irregular versus the normal class present in the dataset.

**- [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#fa7d4f2e58a64fbaaab671105caa85f4):** This dataset contains 2,620 scanned film mammography studies from both CC and MLO views. The labels in this dataset also include benign, malignant, and normal with verified pathology information. We use this dataset only in a testing scenario and qualitatively evaluate the pretrained model on MIAS and INbreast on this data.

## Data Structures
   
```
data 
  └── DATASET Name
        ├── test
        │     ├── normal
        │     │      ├── normal-0.png
        │     │      ├── normal-1.png
        │     │      ├── .
        │     │      ├── .
        │     │      └── .
        │     ├── abnormal
        │     │      ├── mass-0.png
        │     │      ├── mass-1.png
        │     │      ├── .
        │     │      ├── .
        │     │      └── .
        │     └── full image
        │            ├── mask
        │            │      ├── full image 0_mask.png
        │            │      ├── full image 1_mask.png
        │            │      ├── .
        │            │      ├── .
        │            │      └── .
        │            ├── full image 0.png
        │            ├── full image 1.png
        │            ├── .
        │            ├── .
        │            └── .
        └── train
              ├── normal-0.png
              ├── normal-1.png
              ├── .
              ├── .
              └── .
```
          
## Training

To train the model on the MIAS or INBreast datasets with preparing patches and create train and test datasets, run the following:
```
python main.py --dataset=DATASET_NAME --input_height=INPUT_HEIGHT --output_height=OUTPUT_HEIGHT --patch_size=PATCH_SIZE --preparing_data --train
```

## Testing
To evaluate the model on the MIAS or INBreast datasets prepared in train step, run the following:
```
python main.py --dataset=DATASET_NAME --input_height=INPUT_HEIGHT --output_height=OUTPUT_HEIGHT --test
```

To evaluate the generalizability of GAID, we train it on MIAS and INBreast, and test it on the CBIS-DDSM dataset:
```
python main.py --dataset=DATASET_NAME --input_height=INPUT_HEIGHT --output_height=OUTPUT_HEIGHT --test_with_patch=False --test
```


          

## Results
Examples of patches (denoted by X) and their reconstructed versions using AnoGAN, GANomaly ,and GAID.
<p align="center">
  <img src="https://github.com/milad-ahmadi/GAID/blob/master/images/reconstructed results.PNG">
</p>

Testing results of the proposed irregularity detector on the CBIS-DDSM dataset, trained on MIAS and INBreast datasets. Brighter areas of heat-map indicate higher likelihood of irregularity; The heat-map1 and heat-map2 are for training the GAID on MIAS and INBreast datasets, respectively.
<p align="center">
  <img src="https://github.com/milad-ahmadi/GAID/blob/master/images/heat-map results.png">
</p>

## Contact
For questions about our paper or code, please contact [Milad Ahmadi](mailto:milad_ahmadi@comp.iust.ac.ir).

## Acknowledgement
Thanks for @LeeDoYup 's implementation of [AnoGAN] (https://github.com/LeeDoYup/AnoGAN-tf). I implemented GAID based on his implementation.

