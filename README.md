# UPSILoN-T

<div align="center">
<img src="./upsilont/datasets/images/logo.png"><br/>
[ UPSILoN-T ]
</div><br>

UPSILoN-T (UPSILoN Using <b>T</b>ransfer Learning) is the successor to [UPSILoN](https://goo.gl/xmFO6Q). As UPSILoN does, UPSILoN-T aims to classify periodic variable light curves using their time-variability features. The main differences between UPSILoN-T and UPSILoN is as follows:  

 * UPSILoN-T uses Deep Neural Network ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) whereas UPSILoN used [Random Forests](https://en.wikipedia.org/wiki/Random_forest).
    - 2.1% improvement of classification quality ([F1 score](http://en.wikipedia.org/wiki/F1_score)) when compared to the UPSILoN model. 
    - the trained CNN model is significantly smaller than the UPSILoN model (i.e. ~2MB v.s ~4.15GB).

 * UPSILoN-T utilizes [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning).
    - UPSILoN-T allows users to customize the trained CNN model for their own time-variability surveys even if they do not have large-enough dataset. UPSILoN does not offer such functionality.


## Requirement

- PyTorch 1.4+ with CUDA support.
    - https://pytorch.org/
- Scikit-Learn, Numpy, Scipy
    - If not installed, UPSILoN-T will automatically install them.

## Installation

## Test

PyTorch GPU test

```python
import torch

torch.cuda.is_available()
```

Must be True.

UPSILoN-T test

```python
import upsilont
```

## How to Use UPSILoN-T

### Time-Variability Feature Extraction

### Periodic Variable Classification

### Transfer Learning: Finetune the UPSILoN-T Model for Other Surveys


## ChangeLog

#### v0.1.0 (2019/xx/xx)
- release of alpha version
  - provides a pre-trained DNN model.
  - provide modules for feature-extraction, training, prediction and transferring.

## Citation

If you use UPSILoN-T in publication, we would appreciate citations to the paper, [Kim et al. 2020](), which is based on the UPSILoN-T version 0.1.0.


## Contact
Dae-Won Kim, email: dwkim78 at gmail.com

Webpage: https://sites.google.com/site/dwkim78/


#### Keywords

deep learning - transfer learning - classification - astronomy - periodic variables - light curves - variability features - time-series survey
