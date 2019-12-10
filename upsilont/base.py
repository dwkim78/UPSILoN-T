import os
import torch
import pickle
import logging

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from upsilont.component import Net, LightCurveDataset


def apply_log10(x, min_x=None):
    """
    Args:
        x: Input array.
        min_x: (optional) minimum value of 'x'.

    Returns:
        log10(x)
    """
    if min_x is None:
        min_x = np.min(x)

    x -= min_x
    x = x.replace(0., 1e-10)
    x = np.log10(x)

    return x, min_x


class UPSILoNT:
    def __init__(self, device='cuda:0'):
        """
        UPSILoNT class initializer.

        Args:
            device: (optional) GPU device. CPU is not supported.
        """

        # GPU is mandatory. UPSILoN-T does NOT support CPU.
        if not torch.cuda.is_available():
            raise SystemError('Cannot initialize CUDA library. '
                              'Please set your GPU appropriately.')

        # Set device.
        self.device = device

        # Model related parameters.
        self.test_f1 = []
        self.f1_best = 0.
        self.f1_mean = 0.
        self.f1_median = 0.
        self.f1_std = 0.

        self.net = None
        self.n_final = None
        self.min_values = None
        self.norm_params = None
        self.label_encoder = None

        # Set a logger.
        log_format = '%(asctime)s [%(filename)s:%(lineno)s] ' + \
                     '%(levelname)s - %(message)s'
        logging.basicConfig(format=log_format, level=logging.INFO)
        logger = logging.getLogger(__name__)
        self.logger = logger

    def load(self, output_folder: str=None):
        """
        Load a trained UPSILoN model.

        Args:
            output_folder: (optional) A absolute path to a folder
            containing a model and related files.
        """

        # Load the genesis model.
        if output_folder is None:
            output_folder = os.path.join(os.path.dirname(__file__), 'model')

        # Load necessary files.
        self.n_final = pickle.load(open(os.path.join(
            output_folder, 'n_final.pkl'), 'rb'))
        self.min_values = pickle.load(open(os.path.join(
            output_folder, 'min_params.pkl'), 'rb'))
        self.norm_params = pickle.load(open(os.path.join(
            output_folder, 'norm_params.pkl'), 'rb'))
        self.label_encoder = pickle.load(open(os.path.join(
            output_folder, 'label_encoder.pkl'), 'rb'))

        # Load network.
        device = torch.device(self.device)
        net = Net(self.n_final)
        net.to(device)
        genesis_model_path = os.path.join(output_folder, 'state_dict.pt')
        net.load_state_dict(torch.load(genesis_model_path))

        '''
        # Check the number of parameters in each layer.
        for key in torch.load(genesis_model_path).keys():
            if 'bias' in key:
                print(key, torch.load(genesis_model_path)[key].shape)
        '''

        self.net = net

    def predict(self, features: pd.DataFrame, return_prob: bool=False):
        """
        Predict class of input data.

        Args:
            features: Input data.
            prior: (optional) List of prior for each class.
            return_prob: (optional) If True, return a list of
                class probabilities for each sample.
                A list of classes can be accessed by
                "upsilon.label_encoder.classes_"

        Returns:
            Predicted class label.
        """

        # Load a model.
        if self.net is None:
            self.load()

        # Copy features.
        features_in = features.copy()

        # Apply log10 for some features.
        # TODO: fine this code.
        features_in['period'], _ = \
            apply_log10(features_in['period'],
                        self.min_values['min_period'])
        features_in['amplitude'], _ = \
            apply_log10(features_in['amplitude'],
                        self.min_values['min_amplitude'])
        features_in['hl_amp_ratio'], _ = \
            apply_log10(features_in['hl_amp_ratio'],
                        self.min_values['min_hl_amp_ratio'])
        features_in['kurtosis'], _ = \
            apply_log10(features_in['kurtosis'],
                        self.min_values['min_kurtosis'])
        features_in['phase_cusum'], _ = \
            apply_log10(features_in['phase_cusum'],
                        self.min_values['min_phase_cusum'])
        features_in['phase_eta'], _ = \
            apply_log10(features_in['phase_eta'],
                        self.min_values['min_phase_eta'])
        features_in['quartile31'], _ = \
            apply_log10(features_in['quartile31'],
                        self.min_values['min_quartile31'])
        features_in['skewness'], _ = \
            apply_log10(features_in['skewness'],
                        self.min_values['min_skewness'])
        features_in['slope_per90'], _ = \
            apply_log10(features_in['slope_per90'],
                        self.min_values['min_slope_per90'])
        features_in = np.array(features_in)

        # Normalize.
        features_norm = (features_in - self.norm_params[0]) / \
                        self.norm_params[1]

        # Build a dataset with dummy labels.
        labels = np.random.randn(len(features_norm))
        data_set = LightCurveDataset(features_norm, labels)

        # Build data loaders. Do NOT shuffle to keep the order.
        data_loader = torch.utils.data.DataLoader(
            data_set, batch_size=100, shuffle=False, num_workers=2,
            drop_last=False
        )

        predicted_value = []
        predicted_label = []
        predicted_prob = []
        sm = nn.Softmax(dim=1)

        self.net.eval()
        for i, test_data in enumerate(data_loader, 0):
            test_inputs, _ = test_data
            test_inputs = test_inputs.to(self.device)

            outputs = self.net(test_inputs)
            outputs_max = torch.max(outputs, 1)
            outputs_value = outputs_max[0].detach().cpu().numpy()
            outputs_label = outputs_max[1].detach().cpu().numpy()
            predicted_value += outputs_value.tolist()
            predicted_label += outputs_label.tolist()

            if return_prob:
                predicted_prob += sm(outputs).detach().cpu().numpy().tolist()

        # Inverse transform the label (i.e. into string label).
        predicted_label = self.label_encoder.inverse_transform(predicted_label)

        if return_prob:
            return predicted_label, predicted_prob
        else:
            return predicted_label

    def transfer(self, features: pd.DataFrame, labels: pd.DataFrame,
                 output_folder: str, n_iter: int=3, n_epoch: int=50,
                 weight_class: bool=False, balanced_sampling: bool=False,
                 train_last: bool=False, verbose: bool=True):
        """
        Transfer a pre-trained model.

        Args:
            features: Input features to tune a pre-trained model.
            labels: Input labels.
            output_folder: A path to save the transferred
                model and its related files.
            n_iter: (optional) Number of iteration of training.
            n_epoch: (optional) Number of epoch per each training.
            weight_class: (optional) If true, each class is weighted by
                its frequency when calculating loss.
            balanced_sampling: (optional) Choose samples in a balanced manner. If
                True, the number of each class in the selected samples is
                roughly same. Balanced_sample and weight_class cannot be True
                at the same time.
            train_last: (optional) If True, finetune only the last layer.
                If False, finetune all the layers.
            verbose: (optional) Print training information if True.
        """

        # Load the genesis model if none.
        if self.net is None:
            self.load()

        # Transfer.
        self._train(features.copy(), labels.copy(), output_folder,
                    n_iter=n_iter, n_epoch=n_epoch,
                    out_features=np.unique(labels).size,
                    base_net=self.net, weight_class=weight_class,
                    balanced_sampling=balanced_sampling, train_last=train_last,
                    verbose=verbose)

    def train(self, features: pd.DataFrame, labels: pd.DataFrame,
              output_folder: str, n_iter: int=3, n_epoch: int=100,
              out_features: int=None, weight_class: bool=False,
              balanced_sampling: bool=False, verbose: bool=True):
        """
        Train a model from scratch.

        Args:
            features: Input features to tune a pre-trained model.
            labels: Input labels.
            output_folder: A path to save the trained
                model and its related files.
            n_iter: (optional) Number of iteration of training.
            n_epoch: (optional) Number of epoch per each training.
            out_features: (optional) Number of features at the last layer.
            weight_class: (optional) If true, each class is weighted by
                its frequency when calculating loss.
            balanced_sampling: (optional) Choose samples in a balanced manner. If
                True, the number of each class in the selected samples is
                roughly same. Balanced_sample and weight_class cannot be True
                at the same time.
            verbose: (optional) Print training information if True.
        """

        self._train(features.copy(), labels.copy(), output_folder,
                    n_iter=n_iter, n_epoch=n_epoch, out_features=out_features,
                    balanced_sampling=balanced_sampling,
                    weight_class=weight_class,
                    verbose=verbose)

    def _get_balanced_sample_weights(self, labels):
        """
        Return weights for oversampling of imbalanced dataset.

        Args:
            labels: Labels encoded (i.e. integer numbers)

        Returns:
            Weights by class size.
        """

        unique, counts = np.unique(labels, return_counts=True)
        counts = np.array(counts)
        rev_counts = 1. / counts
        normed_weights = rev_counts / np.sum(rev_counts)
        samples_weight = normed_weights[labels]
        weights = torch.FloatTensor(samples_weight).to(self.device)

        return weights

    def _train(self, features: pd.DataFrame, labels: pd.DataFrame,
               output_folder: str, n_iter: int=3, n_epoch: int=100,
               out_features: int=None, weight_class: bool=False,
               balanced_sampling: bool = False,
               base_net: Net=None, train_last: bool=False,
               verbose: bool=True):
        """
        Train a model either from scratch or using transfer learning.

        Args:
            features: Input features to tune a pre-trained model.
            labels: Input labels.
            output_folder: A path to save the model and its related files.
            n_iter: (optional) Number of iteration of training.
            n_epoch: (optional) Number of epoch per each training.
            out_features: (optional) Number of features at the last layer.
            weight_class: (optional) If true, each class is weighted by
                its frequency when calculating loss.
            balanced_sampling: (optional) Choose samples in a balanced manner. If
                True, the number of each class in the selected samples is
                roughly same. Balanced_sample and weight_class cannot be True
                at the same time.
            base_net: (optional) Only used when doing transfer learning.
            train_last: (optional) If True, finetune only the last layer.
                If False, finetune all the layers.
            verbose: (optional) Print training information if True.
        """

        # weight_class and balanced_sample cannot be True at the same time.
        # if weight_class and balanced_sample:
        #     raise ValueError('weight_class and balanced_sample cannot be '
        #                      '"True" at the same time.')

        # Make an output folder if not exist.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # apply log10 to some features.
        # TODO: fine this code.
        features['period'], min_period = apply_log10(features['period'])
        features['amplitude'], min_amplitude = apply_log10(features['amplitude'])
        features['hl_amp_ratio'], min_hl_amp_ratio = \
            apply_log10(features['hl_amp_ratio'])
        features['kurtosis'], min_kurtosis = apply_log10(features['kurtosis'])
        features['phase_cusum'], min_phase_cusum = \
            apply_log10(features['phase_cusum'])
        features['phase_eta'], min_phase_eta = \
            apply_log10(features['phase_eta'])
        features['quartile31'], min_quartile31 = \
            apply_log10(features['quartile31'])
        features['skewness'], min_skewness = apply_log10(features['skewness'])
        features['slope_per90'], min_slope_per90 = \
            apply_log10(features['slope_per90'])

        min_values = {
            'min_period': min_period,
            'min_amplitude': min_amplitude,
            'min_hl_amp_ratio': min_hl_amp_ratio,
            'min_kurtosis': min_kurtosis,
            'min_phase_cusum': min_phase_cusum,
            'min_phase_eta': min_phase_eta,
            'min_quartile31': min_quartile31,
            'min_skewness': min_skewness,
            'min_slope_per90': min_slope_per90
        }

        self.min_values = min_values
        # Save for later usage.
        pickle.dump(self.min_values, open(os.path.join(
            output_folder, 'min_params.pkl'), 'wb'))

        features = np.array(features)
        labels = np.array(labels)

        # Normalize.
        features_median = np.median(features, axis=0)
        features_std = np.std(features, axis=0)
        features_norm = (features - features_median) / features_std

        # Save the number of features at the last layers.
        if out_features is None:
            self.n_final = np.unique(labels).size
        else:
            self.n_final = out_features

        # Save.
        pickle.dump(self.n_final, open(os.path.join(
            output_folder, 'n_final.pkl'), 'wb'))

        # Save the values for later usage (e.g. prediction).
        self.norm_params = [features_median, features_std]
        pickle.dump(self.norm_params, open(os.path.join(
            output_folder, 'norm_params.pkl'), 'wb'))

        # Fit a label encoder.
        le = LabelEncoder()
        le.fit(labels)
        labels_encoded = le.transform(labels)

        # Save the label encoder.
        self.label_encoder = le
        pickle.dump( self.label_encoder, open(os.path.join(
            output_folder, 'label_encoder.pkl'), 'wb'))

        # Derive class weight by its frequency.
        if weight_class:
            unique, counts = np.unique(labels_encoded, return_counts=True)
            counts = np.array(counts)
            rev_counts = 1. / counts
            weights = rev_counts / np.sum(rev_counts)
            class_weights = torch.FloatTensor(weights).to(self.device)

        # Training information.
        training_info = {'learning_rate': [], 'training_loss': [],
                         'validation_loss': [], 'test_f1': [],
                         'training_f1': []}

        # Train a model for the number of iteration.
        best_f1 = 0.
        f1_average = 'macro'
        for i in range(n_iter):
            # Train and test set split. So each iteration,
            # using a set separated differently.
            x_train, x_test, y_train, y_test = \
                train_test_split(features_norm, labels_encoded,
                                 test_size=0.2, stratify=labels_encoded)

            # Build datasets.
            trainset = LightCurveDataset(x_train, y_train)
            testset = LightCurveDataset(x_test, y_test)

            # Up-sampling imbalanced dataset.
            if balanced_sampling:
                train_weights = self._get_balanced_sample_weights(y_train)
                test_weights = self._get_balanced_sample_weights(y_test)

                train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    train_weights, len(train_weights), replacement=True)
                test_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    test_weights, len(test_weights), replacement=True)
                shuffle = False
            else:
                train_sampler = None
                test_sampler = None
                shuffle = True

            # Build data loaders.
            batch_size = 1024
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=shuffle,
                sampler=train_sampler, num_workers=2)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=shuffle,
                sampler=test_sampler, num_workers=2)

            # Initialize a network before entering the iteration.
            net = Net()
            if base_net is not None:
                # For transfer learning.
                net.load_state_dict(base_net.state_dict())

            # Set the number of neurons at the final layers, which is
            # actually the number of target classes.
            net.fc4 = nn.Linear(net.bn4.num_features, self.n_final)
            net.bn5 = nn.BatchNorm1d(self.n_final)

            # Use a specific device.
            net.to(self.device)

            # Initial learning rate.
            learning_rate = 0.1

            # Set training instances.
            if base_net is not None:
                # Transfer only the last layer.
                if train_last:
                    optimizer = optim.SGD(net.fc4.parameters(), lr=learning_rate,
                                          momentum=0.9)
                else:
                    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                                          momentum=0.9)
            else:
                optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                                      momentum=0.9)

            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3,
                                          eps=1e-15)
            if weight_class:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()

            # Iterate.
            for epoch in range(n_epoch):
                running_loss = 0.0

                # Iterate learning rate.
                if optimizer.param_groups[0]['lr'] <= 1e-10:
                    optimizer.param_groups[0]['lr'] = learning_rate

                # For each batch.
                predicted_label = []
                true_label = []
                net.train()
                for l, data in enumerate(trainloader, 0):
                    # Get the inputs.
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), \
                                     labels.to(self.device)

                    # Zero the parameter gradients.
                    optimizer.zero_grad()

                    # Forward + backward + optimize.
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    # Get true and predicted labels.
                    outputs_numpy = torch.max(outputs, 1)[1].cpu().numpy()
                    test_numpy = labels.cpu().numpy()
                    predicted_label += outputs_numpy.tolist()
                    true_label += test_numpy.tolist()

                    loss.backward()
                    optimizer.step()

                    # Running loss.
                    running_loss += loss.item()

                # Calculate training f1.
                training_f1 = matthews_corrcoef(true_label, predicted_label,
                                       average=f1_average)

                # Get test-set performance
                val_loss = 0.
                predicted_label = []
                true_label = []
                net.eval()
                for m, test_data in enumerate(testloader, 0):
                    test_inputs, test_labels = test_data
                    test_inputs, test_labels = test_inputs.to(self.device), \
                                               test_labels.to(self.device)

                    outputs = net(test_inputs)
                    val_loss += criterion(outputs, test_labels).item()

                    # Get true and predicted labels.
                    outputs_numpy = torch.max(outputs, 1)[1].cpu().numpy()
                    test_numpy = test_labels.cpu().numpy()
                    predicted_label += outputs_numpy.tolist()
                    true_label += test_numpy.tolist()

                test_f1 = matthews_corrcoef(true_label, predicted_label,
                                   average=f1_average)
                curr_f1 = test_f1

                if verbose:
                    self.logger.info(('[{0}, {1}] '
                                      'train F1: {2:.6f}, test F1: {3:.6f}, '
                                      'learning rate {4:.1e}').format(
                          i + 1, epoch + 1, training_f1, curr_f1,
                          optimizer.param_groups[0]['lr'])
                    )

                # Save training information for later usage.
                training_info['learning_rate'].append(
                    optimizer.param_groups[0]['lr'])
                training_info['training_loss'].append(running_loss)
                training_info['validation_loss'].append(val_loss)
                training_info['training_f1'].append(training_f1)
                training_info['test_f1'].append(curr_f1)

                pickle.dump(training_info, open(os.path.join(
                    output_folder, 'training_info.pkl'), 'wb'))

                # Only if the new model is better.
                if curr_f1 > best_f1:
                    best_f1 = curr_f1
                    self.f1_best = best_f1

                    # Save the model.
                    torch.save(net.state_dict(), os.path.join(
                        output_folder, 'state_dict.pt'))
                    self.net = net
                    # self.logger.info('Better model saved.')

                    # Save true and predicted labels for later usages.
                    pickle.dump([true_label, predicted_label],
                                open(os.path.join(output_folder,
                                                  'true_predicted.pkl'), 'wb'))

                    # Save the best F1 as a plain text for temporary saving.
                    fp = open(os.path.join(output_folder, 'info.txt'), 'w')
                    fp.writelines('# F1: {0:.6f}\n'.format(best_f1))
                    fp.close()

                # Scheduler based on validation loss (i.e. test-set loss).
                scheduler.step(val_loss)

            # Epoch ends.
            if verbose:
                self.logger.info('The overall best F1 using the '
                                 'validation set: {0:.6f}'.format(self.f1_best))

        ################################
        # The whole training finishes. #
        ################################

        # Get the best test F1 for each iteration.
        test_f1 = np.max(
            np.array(training_info['test_f1']).reshape(-1, n_epoch), axis=1)
        self.test_f1 = test_f1
        # Calculate statistics of test_f1.
        self.f1_mean = np.mean(test_f1)
        self.f1_median = np.median(test_f1)
        self.f1_std = np.std(test_f1)

        # Save F1 information.
        fp = open(os.path.join(output_folder, 'info.txt'), 'w')
        fp.writelines('# Best_F1 Median_F1 Mean_F1 Std_F1\n')
        fp.writelines('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(
            self.f1_best, self.f1_median, self.f1_mean, self.f1_std))
        fp.close()

    # TODO: Comment this out before package release.
    def _get_feature_importance(self, features: pd.DataFrame,
                                labels: pd.DataFrame):
        """Test purpose only."""

        import shap

        # Load a model.
        if self.net is None:
            self.load()

        # Copy features.
        features_in = features.copy()

        # Apply log10 for some features.
        # TODO: fine this code.
        features_in['period'], _ = \
            apply_log10(features_in['period'],
                        self.min_values['min_period'])
        features_in['amplitude'], _ = \
            apply_log10(features_in['amplitude'],
                        self.min_values['min_amplitude'])
        features_in['hl_amp_ratio'], _ = \
            apply_log10(features_in['hl_amp_ratio'],
                        self.min_values['min_hl_amp_ratio'])
        features_in['kurtosis'], _ = \
            apply_log10(features_in['kurtosis'],
                        self.min_values['min_kurtosis'])
        features_in['phase_cusum'], _ = \
            apply_log10(features_in['phase_cusum'],
                        self.min_values['min_phase_cusum'])
        features_in['phase_eta'], _ = \
            apply_log10(features_in['phase_eta'],
                        self.min_values['min_phase_eta'])
        features_in['quartile31'], _ = \
            apply_log10(features_in['quartile31'],
                        self.min_values['min_quartile31'])
        features_in['skewness'], _ = \
            apply_log10(features_in['skewness'],
                        self.min_values['min_skewness'])
        features_in['slope_per90'], _ = \
            apply_log10(features_in['slope_per90'],
                        self.min_values['min_slope_per90'])
        features_in = np.array(features_in)

        # Normalize.
        features_norm = (features_in - self.norm_params[0]) / \
                         self.norm_params[1]
        column_names = features.columns.values.tolist()

        device = 'cuda:0'
        labels_encoded = self.label_encoder.transform(np.array(labels))

        x_train, x_test, y_train, y_test = \
            train_test_split(features_norm, labels_encoded,
                             test_size=0.2, stratify=labels_encoded)

        # Build datasets.
        trainset = LightCurveDataset(x_train, y_train)
        testset = LightCurveDataset(x_test, y_test)

        # Up-sampling imbalanced dataset.
        train_weights = self._get_balanced_sample_weights(y_train)
        test_weights = self._get_balanced_sample_weights(y_test)

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            train_weights, len(train_weights), replacement=True)
        test_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            test_weights, len(test_weights), replacement=True)
        shuffle = False

        # Build data loaders.
        train_batch_size = 20000
        test_batch_size = 2000
        # train_batch_size = 100
        # test_batch_size = 10
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=shuffle,
            sampler=train_sampler, num_workers=1)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=shuffle,
            sampler=test_sampler, num_workers=1)

        for l, data in enumerate(trainloader, 0):
            x_train_samples, _ = data
            x_train_samples = x_train_samples.to(device)
            break

        for l, data in enumerate(testloader, 0):
            x_samples, _ = data
            x_samples = x_samples.to(device)
            break

        x_train_samples = x_train_samples.reshape(-1, 16)
        x_samples = x_samples.reshape(-1, 16)

        # x_train_samples = torch.FloatTensor(
        #     x_train[np.random.choice(np.arange(len(x_train)), 100,
        #                              replace=False)]
        # ).to(device)
        e = shap.DeepExplainer(self.net, x_train_samples)

        # x_samples = torch.FloatTensor(
        #     x_train[np.random.choice(np.arange(len(x_train)), 10,
        #                              replace=False)]
        # ).to(device)
        # print(x_train_samples)
        # print(x_samples)

        column_converter = {
            'amplitude': r'$H_1$',
            'hl_amp_ratio': r'$A$',
            'kurtosis': r'$\gamma_2$',
            'period': r'Period',
            'phase_cusum': r'$\psi^{CS}$',
            'phase_eta': r'$\psi^\eta$',
            'phi21': r'$\phi_{21}$',
            'phi31': r'$\phi_{31}$',
            'quartile31': r'$Q_{3-1}$',
            'r21': r'$R_{21}$',
            'r31': r'$R_{31}$',
            'shapiro_w': r'$W$',
            'skewness': r'$\gamma_1$',
            'slope_per10': r'$m_{p10}$',
            'slope_per90': r'$m_{p90}$',
            'stetson_k': r'$K$'
        }
        for i in range(len(column_names)):
            for key, value in column_converter.items():
                if column_names[i] == key:
                    column_names[i] = value

        labels = [ele.replace('_', ' ') for ele in self.label_encoder.classes_]
        shap_values = e.shap_values(x_samples)

        from upsilont.plot.shap_plot import summary_plot

        pl = summary_plot(shap_values, features=x_samples,
                          feature_names=column_names,
                          class_names=labels)

        # pl.show()
        pl.savefig('/home/kim/Temp/SHAP.eps')
