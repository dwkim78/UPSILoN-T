import os
import torch
import pickle
import logging

import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from upsilont.component import Net, LightCurveDataset


class UPSILoNT:
    def __init__(self, device='cuda:0'):
        # GPU is mandatory. UPSILoN-T does NOT support CPU.
        if not torch.cuda.is_available():
            raise SystemError('Cannot initialize CUDA library. '
                              'Please set your GPU appropriately.')

        # Set device.
        self.device = device

        # Model related parameters.
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
        # Load the genesis model.
        if output_folder is None:
            output_path = os.path.join(os.path.dirname(__file__), 'model')

        # Load necessary files.
        self.n_final = pickle.load(open(os.path.join(
            output_path, 'n_final.pkl'), 'rb'))
        self.min_values = pickle.load(open(os.path.join(
            output_path, 'min_params.pkl'), 'rb'))
        self.norm_params = pickle.load(open(os.path.join(
            output_path, 'norm_params.pkl'), 'rb'))
        self.label_encoder = pickle.load(open(os.path.join(
            output_path, 'label_encoder.pkl'), 'rb'))

        # Load network.
        device = torch.device(self.device)
        net = Net(self.n_final)
        net.to(device)
        genesis_model_path = os.path.join(output_path, 'state_dict.pt')
        net.load_state_dict(torch.load(genesis_model_path))

        self.net = net

    def predict(self, features):
        # Load a model.
        if self.net is None:
            self.load()

        # Normalize.
        features_norm = (features - self.norm_params[0]) / self.norm_params[1]

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
        self.net.eval()
        for i, test_data in enumerate(data_loader, 0):
            test_inputs, _ = test_data
            test_inputs = test_inputs.to(self.device)

            outputs = self.net(test_inputs)
            outputs_max = torch.max(outputs, 1)
            outputs_value = outputs_max[0].cpu().numpy()
            outputs_label = outputs_max[1].cpu().numpy()
            predicted_value += outputs_value.tolist()
            predicted_label += outputs_label.tolist()

        # Inverse transform the label (i.e. into string label).
        predicted_label = self.label_encoder.inverse_transform(predicted_label)

        return predicted_label

    def save(self):
        self.net

    def preprocess(self):
        pass

    def transfer(self, features: np.ndarray, labels: np.ndarray,
                 output_folder: str, n_iter: int=3, n_epoch: int=100):
        # Load the genesis model.
        self.load()

        # Transfer.
        self.train(features, labels, output_folder, n_iter, n_epoch,
                   np.unique(labels).size, self.net)

    def train(self, features: np.ndarray, labels: np.ndarray,
              output_folder: str, n_iter: int=10, n_epoch: int=500,
              out_features: int=None, base_net: Net=None):

        # Get values for normalization.
        features_median = np.median(features, axis=0)
        features_std = np.std(features, axis=0)

        # Normalize.
        features_norm = (features - features_median) / features_std

        # Make an output folder if not exist.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

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

        # Training information.
        training_info = {'learning_rate': [], 'training_loss': [],
                         'validation_loss': [], 'test_f1': [],
                         'training_f1': []}

        # Train a model for the number of iteration.
        best_f1 = 0.
        for i in range(n_iter):
            # Train and test set split. So each iteration,
            # using a set separated differently.
            x_train, x_test, y_train, y_test = \
                train_test_split(features_norm, labels_encoded,
                                 test_size=0.1, stratify=labels_encoded)

            # Build datasets.
            trainset = LightCurveDataset(x_train, y_train)
            testset = LightCurveDataset(x_test, y_test)

            # Build data loaders.
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=1024, shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=1024, shuffle=True, num_workers=2)

            # Initialize a network before entering the iteration.
            net = Net()
            if base_net is not None:
                # For transfer learning.
                net.load_state_dict(base_net.state_dict())

            # Use a specific device.
            net.to(self.device)

            # Set the number of neurons at the final layers, which is
            # actually the number of target classes.
            net.fc3.out_features = self.n_final
            net.bn3.num_features = self.n_final

            # Initial learning rate.
            learning_rate = 0.1

            # Set training instances.
            optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                                  momentum=0.9)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3,
                                          eps=1e-15)
            criterion = nn.CrossEntropyLoss()

            # Iterate.
            for epoch in range(n_epoch):
                running_loss = 0.0

                # Iterate learning rate
                if optimizer.param_groups[0]['lr'] <= 1e-14:
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
                training_f1 = f1_score(true_label, predicted_label,
                                       average='weighted')

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

                f1_weighted = f1_score(true_label, predicted_label,
                                       average='weighted')
                curr_f1 = f1_weighted

                self.logger.info(('[{0}, {1}] '
                                  'train F1: {2:.6f}, test F1: {3:.6f}, '
                                  'learning rate {4:.1e}').format(
                      i + 1, epoch + 1,
                      training_f1, curr_f1, optimizer.param_groups[0]['lr']))

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

                    # Save the best F1 as a plain text.
                    fp = open(os.path.join(output_folder, 'info.txt'), 'w')
                    fp.writelines('# F1: {0:.6f}\n'.format(best_f1))
                    fp.close()

                # Scheduler based on validation loss (i.e. test-set loss).
                scheduler.step(val_loss)

            # Epoch ends.
            self.logger.info('The current best F1: {0:.6f}'.format(
                self.f1_best))

        ################################
        # The whole training finishes. #
        ################################

        # Get the best test F1 for each iteration.
        test_f1 = np.max(
            np.array(training_info['test_f1']).reshape(-1, n_epoch), axis=1)
        self.f1_mean = np.mean(test_f1)
        self.f1_median = np.median(test_f1)
        self.f1_std = np.std(test_f1)
