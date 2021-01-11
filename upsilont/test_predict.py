import time
import pandas as pd

from upsilont import UPSILoNT
from upsilont.features import VariabilityFeatures
from upsilont.datasets import load_lc
from upsilont import Logger


def run():
    """Test UPSILoN-T package."""

    date, mag, err = load_lc()

    index = mag < 99.999
    date = date[index]
    mag = mag[index]
    err = err[index]

    logger = Logger().getLogger()
    logger.info('Extract variability features.')

    feature_list = []
    for i in range(1):
        start = time.time()
        var_features = VariabilityFeatures(date, mag, err)
        features = var_features.get_features()
        # for key, value in features.items():
        #     print('   %s: %f' % (key, value))
        feature_list.append(features)

    logger.info('Convert to Pandas DataFrame.')
    pd_features = pd.DataFrame(feature_list)

    logger.info('Predict using UPSILoN-T.')
    ut = UPSILoNT(logger=logger)
    label, prob = ut.predict(pd_features, return_prob=True)

    logger.info('Predicted class: {0}'.format(label))
    logger.info('Probability: {0}'.format(prob))
    logger.info('Corresponding classes: {0}'.format(ut.label_encoder.classes_))
    logger.info('Done.')

    logger.handlers = []


if __name__ == '__main__':
    run()
