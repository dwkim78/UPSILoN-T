# UPSILoN-T
<div align="center">
<img src="./upsilont/datasets/images/logo.png"><br/>
[ UPSILoN-T ]
</div><br>


UPSILoN-T aims to classify periodic variable stars using deep learning and transfer learning.


## Index

1. [Installation](#1-installation)
2. [How to Use UPSILoN-T](#2-usage)


## 1. Installation

- Register for the ETRI OSS GitLab repository, https://etrioss.kr/
- Ask permission to D.-W. Kim (dwk@etri.er.kr) or D. Yeo (yeody@etri.re.kr)
- Access to https://etrioss.kr/ksb/upsilon-t
- Install the package as follows.

```python
pip install git+https://etrioss.kr/ksb/upsilon-t
```


## 2. Usage

You can test UPSILoN-T package as follow.

```python
from upsilon import test_predict

test_predict()
```

The results should look like:

```sh
yyyy-mm-dd hh:mm:ss,sss [test.py:21] INFO - Extract variability features.
yyyy-mm-dd hh:mm:ss,sss [test.py:32] INFO - Convert to Pandas DataFrame.
yyyy-mm-dd hh:mm:ss,sss [test.py:35] INFO - Predict Using UPSILoN-T.
yyyy-mm-dd hh:mm:ss,sss [test.py:39] INFO - Predicted Class: ['RRL_ab']
yyyy-mm-dd hh:mm:ss,sss [test.py:40] INFO - Probability: [[1.4387337614607532e-05, 0.0003274077898822725, 0.001863368903286755, 1.1310783520457335e-05, 1.27751472973614e-05, 0.004155108239501715, 0.009323867969214916, 0.019384946674108505, 0.0017969163600355387, 5.909596802666783e-05, 6.260129794100067e-06, 1.8606748426464037e-06, 8.244837954407558e-05, 7.316847768379375e-05, 0.0005118269473314285, 5.3744974138680845e-05, 0.9620862603187561, 6.40879588900134e-05, 0.000131708788103424, 3.8427057006629184e-05, 1.0599986808301765e-06]]
yyyy-mm-dd hh:mm:ss,sss [test.py:41] INFO - Corresponding classes: ['BV' 'CEPH_1O' 'CEPH_F' 'CEPH_Other' 'DSCT' 'EB_EC' 'EB_ED' 'EB_ESD'
 'LPV_Mira_AGB_C' 'LPV_Mira_AGB_O' 'LPV_OSARG_AGB' 'LPV_OSARG_RGB'
 'LPV_SRV_AGB_C' 'LPV_SRV_AGB_O' 'NonVar' 'QSO' 'RRL_ab' 'RRL_c' 'RRL_d'
 'RRL_e' 'T2CEPH']
yyyy-mm-dd hh:mm:ss,sss [test.py:42] INFO - Done.
```

### 2.1 Predict Variable Classes

### 2.2 Transfer UPSILoN-T

### 2.3 Dealing with Imbalanced Datasets

## Citation

If you use UPSILoN-T in publication, we would appreciate citations to the paper, <a href="http://">in progress</a>.

