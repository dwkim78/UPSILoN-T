# UPSILoN-T
<div align="center">
<img src="./upsilont/datasets/images/logo.png"><br/>
[ UPSILoN-T ]
</div><br>


UPSILoN-T aims to classify periodic variable stars using deep learning and transfer learning.


## Index

1. [Dependency](#1-dependency)
2. [Installation](#2-installation)
3. [How to Use UPSILoN-T](#3-usage)


## 1. Dependency
[Python 3.6+](https://www.python.org/)

[PyTorch 1.6+](https://pytorch.org/)

[Numpy 1.17+](http://www.numpy.org/)

[Scipy 1.3+](http://scipy.org/)

[scikit-learn 0.22](http://scikit-learn.org/stable/)

[pyFFTW 0.10.4+](http://hgomersall.github.io/pyFFTW/)

 * pyFFTW is optional but <b>highly recommended</b> for multi-threads usage for FFT. You will need to install [FFTW](http://www.fftw.org/) as well prior to installation of pyFFTW.

These libraries (except pyFFTW) will be automatically installed if your machines do not have them yet. In the case you encounter errors during the installation of the dependencies, try to install them individually. Your machine may not have other necessary libraries that are required by the dependencies.

In addition, GPU is required to use the UPSILoN-T package.


## 2. Installation

```shell script
  pip install git+https://github.com/dwkim78/UPSILoN-T
```

## 3. Usage

You can test UPSILoN-T package as follow.

```python
from upsilont import test_predict

test_predict()
```

The results should look like:

```sh
yyyy-mm-dd hh:mm:ss,sss [test.py:21] INFO - Extract variability features.
yyyy-mm-dd hh:mm:ss,sss [test.py:32] INFO - Convert to Pandas DataFrame.
yyyy-mm-dd hh:mm:ss,sss [test.py:35] INFO - Predict using UPSILoN-T.
yyyy-mm-dd hh:mm:ss,sss [test.py:39] INFO - Predicted class: ['RRL_ab']
yyyy-mm-dd hh:mm:ss,sss [test.py:40] INFO - Probability: [[1.4387337614607532e-05, 0.0003274077898822725, 0.001863368903286755, 1.1310783520457335e-05, 1.27751472973614e-05, 0.004155108239501715, 0.009323867969214916, 0.019384946674108505, 0.0017969163600355387, 5.909596802666783e-05, 6.260129794100067e-06, 1.8606748426464037e-06, 8.244837954407558e-05, 7.316847768379375e-05, 0.0005118269473314285, 5.3744974138680845e-05, 0.9620862603187561, 6.40879588900134e-05, 0.000131708788103424, 3.8427057006629184e-05, 1.0599986808301765e-06]]
yyyy-mm-dd hh:mm:ss,sss [test.py:41] INFO - Corresponding classes: ['BV' 'CEPH_1O' 'CEPH_F' 'CEPH_Other' 'DSCT' 'EB_EC' 'EB_ED' 'EB_ESD'
 'LPV_Mira_AGB_C' 'LPV_Mira_AGB_O' 'LPV_OSARG_AGB' 'LPV_OSARG_RGB'
 'LPV_SRV_AGB_C' 'LPV_SRV_AGB_O' 'NonVar' 'QSO' 'RRL_ab' 'RRL_c' 'RRL_d'
 'RRL_e' 'T2CEPH']
yyyy-mm-dd hh:mm:ss,sss [test.py:42] INFO - Done.
```

### 2.1 Predict Variable Classes

The following pseudocode shows how to use the package to extract variability features from a set of light-curves and then to predict their variable classes.

```python
from upsilont import UPSILoNT
from upsilont.features import VariabilityFeatures

# Extract features from a set of light-curves. 
feature_list = []
for light_curve in set_of_light_curves:

    # Read a light-curve.
    date = np.array([:])
    mag = np.array([:])
    err = np.array([:])
         
    # Extract features.
    var_features = VariabilityFeatures(date, mag, err).get_features()
    
    # Append to the list.
    feature_list.append(var_features)

# Convert to Pandas DataFrame.
features = pd.DataFrame(feature_list)
    
# Classify.
ut = UPSILoNT()
label, prob = ut.predict(features, return_prob=True)
```
```label``` and ```prob``` is a list of predicted classes and class probabilities, respectively. 

### 2.2 Transfer UPSILoN-T

```python
# Get features and labels.
features = ...
labels = ...

# Transfer UPSILoN-T.
ut = UPSILoNT()
ut.transfer(features, labels, "/path/to/transferred/model/")
```

```features``` is a list of features extracted from a set of light-curves, and ```labels``` is a list of corresponding labels. The package writes the transferred model and other model-related parameters to a specified location (i.e. ```/path/to/transferred/model/``` in the above pseudocode). You can load the transferred model:

```python
ut = UPSILoNT()
ut.load("/path/to/transferred/model/")
```

The loaded model either can be used for prediction or can be transferred.

### 2.3 Dealing with Imbalanced Datasets

The UPSILoN-T library provides two approaches dealing with an imbalanced dataset, one is weighting a loss function and another is over-sampling. You can use these approaches as follows:

```python
# Over-sampling.
ut.train(features, labels, balanced_sampling=True)

# Weighting a loss function.
ut.train(features, labels, weight_class=True)

# Do both.
ut.train(features, labels, balanced_sampling=True, weight_class=True)
```

For transferring a model, you can use ```ut.transfer``` rather than ```ut.train```.


### Note

To get the latest version of UPSILoN-T, you might want to visit [ETRI GitLab](https://etrioss.kr/ksb/upsilon-t).

1. Register for the ETRI OSS GitLab website, https://etrioss.kr/
2. Contact D.-W. Kim (dwk@etri.er.kr) for access permission.
3. Check if you can access to https://etrioss.kr/ksb/upsilon-t
4. Install the package as follows.

  ```shell script
  pip install git+https://etrioss.kr/ksb/upsilon-t
   ```

## Citation

If you use UPSILoN-T in publication, we would appreciate citations to the paper, <a href="https://arxiv.org/abs/2106.00187">Kim et al. 2021</a>.

