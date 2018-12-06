import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pickle, gzip
import glob
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn, optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

train = 0

##### TRAIN ###
if train == 1:
    train = pd.read_csv('training_set.csv')
    meta_train = pd.read_csv('training_set_metadata.csv')
    train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
    train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']

#Aggregate features can enable ease of feature extraction approaches.
#
#
    aggs = {
        'mjd': ['min', 'max', 'size'],
        'passband': ['min', 'max', 'mean', 'median', 'std'],
        'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum','skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

    agg_train = train.groupby('object_id').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_train.columns = new_columns
    agg_train['mjd_diff'] = agg_train['mjd_max'] - agg_train['mjd_min']
    agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
    agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
    agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
    agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']
    agg_train['flux_deriv_time'] = agg_train['flux_diff'] / agg_train['mjd_diff']
    del agg_train['mjd_max'], agg_train['mjd_min']

# Following block follows as a time series data.
#
#
    time_series_data = []
    time_series_data_shapes = []
    rows = [1, 2, 3, 4, 5, 6, 7]
    rows_clipped = [0, 1, 2, 3, 4, 5, 6]
    unique_ids = sorted(train['object_id'].unique())
    id_group = train.groupby(['object_id'])
    for id in unique_ids:
        current_id_group = id_group.get_group(id)
        new_id_sample = current_id_group.transpose().as_matrix(columns=current_id_group.transpose().columns[1:])
        new_id_sample = np.array(new_id_sample)
        
        # Problem with this approach is not the same number of recordings are taken for each sample. Have to take subsets of data
        # in which information could get lost.
        
        # Take sample sizes of the original data
        columns = list(range(0, new_id_sample.shape[1], int(new_id_sample.shape[1]/46)))
        next_sample = new_id_sample[np.array(rows)[:, None], np.array(columns)]
        next_sample_clipped = next_sample[np.array(rows_clipped)[:, None], np.array(list(range(46)))]
        next_sample_clipped = np.array(next_sample_clipped).transpose()
        sample_final = np.expand_dims(next_sample_clipped, axis=0) #Expand dimensions for torch input.
        
        time_series_data.append(sample_final)

# Merge training metadata with aggregate data.
    full_train = agg_train.reset_index().merge(
        right=meta_train,
        how='outer',
        on='object_id'
    )

# Collect object_ids
    object_ids = full_train['object_id']
    del full_train['object_id']

# Extract classes
    if 'target' in full_train:
        y = full_train['target']
        del full_train['target']
    classes = sorted(y.unique())

    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    class_weight = {
        c: 1 for c in classes
    }
    for c in [64, 15]:
        class_weight[c] = 2

    print('Unique classes : ', classes)

# Create a one-hot encoding of the target data
    def encode_one_hot(targ):
        s = pd.Series(classes)
        one_hot = s.isin([targ])
        dfList = one_hot.tolist()
        return np.array(dfList).astype(int)

# Expand dimensions of classes for torch.
    classifications = np.array([np.expand_dims(np.expand_dims(encode_one_hot(targ), axis=0), axis=2) for targ in y])

# Fill in NaN cells with the average of the columns
    for j in range(full_train.shape[1]):
            full_train.iloc[:,j]=full_train.iloc[:,j].fillna(full_train.iloc[:,j].mean())

def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    #y_ohe = y_ohe.detach().numpy()
    y_p = y_p.cpu()
    y_ohe = y_ohe.cpu()
    y_p = y_p.detach().numpy()
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    y_p_log = torch.Tensor(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = torch.sum(torch.mul(y_ohe, y_p_log))
    # Get the number of positives for each class
    nb_pos = torch.sum(y_ohe).float()
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    class_arr = torch.Tensor(class_arr)
    y_w = y_log_ones * class_arr / nb_pos
    loss = - torch.sum(y_w) / torch.sum(class_arr)
    return loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def accuracy(y_true, y_pred):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    y_pred = y_pred.detach().numpy()
    y_diff = abs(y_pred-y_true)
    accuracy = y_diff.sum(1).mean()
    return 1.0 - accuracy


# Convolutional Network
ndf = 64
nc = 1
n_classes = 14

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.main = nn.Sequential(
                                  # input is 1 x 46 x 7
                                  nn.Conv2d(in_channels=nc, out_channels=50, kernel_size=(4, 1), stride=(2, 1) ,padding = 1, bias=True),
                                  nn.Tanh(),
                                  # state size. (50) x 23 x 7
                                  nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(4, 1), stride=1, bias=True),
                                  nn.BatchNorm2d(50),
                                  nn.Tanh(),
                                  # state size. (50) x 20 x 7
                                  nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(4, 1), stride=1 ,bias=True),
                                  nn.BatchNorm2d(50),
                                  nn.Tanh(),
                                  # state size. (50) x 17 x 7
                                  nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(4, 1), stride=1 ,bias=True),
                                  nn.BatchNorm2d(50),
                                  nn.Tanh(),
                                  # state size. (50) x 14 x 9
                                  nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 4), stride=1 ,bias=True),
                                  nn.BatchNorm2d(50),
                                  nn.Tanh(),
                                  # state size. (50) x 14 x 6
                                  nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 4), stride=1 ,bias=True),
                                  nn.BatchNorm2d(50),
                                  nn.Tanh(),
                                  # state size. (50) x 14 x 3
                                  nn.Conv2d(in_channels=50, out_channels=1, kernel_size=(1, 3), stride=1 ,bias=True),
                                  nn.Sigmoid()
                                  # state size (50, 14, 1)
                                  )

    def forward(self, input):
        return self.main(input)

convolutionalNet = ConvNet()
if torch.cuda.is_available() and train == 1:
    convolutionalNet.cuda()

optimizer = optim.Adam(convolutionalNet.parameters(), lr=0.0002)

def train_simple(optimizer, inputs, class_indices):
    optimizer.zero_grad()
    prediction = convolutionalNet(inputs)
    error = multi_weighted_logloss(prediction, class_indices)
    error.backward()
    optimizer.step()
    return error

class plasDataset(Dataset):
  def __init__(self, samples, labels):
        self.len = samples.shape[0]
        self.y = labels
        self.x = samples

  def __getitem__(self, index):
        return self.x[index], self.y[index]

  def __len__(self):
        return self.len

if train == 1:
    time_series_data = np.array(time_series_data)
    training_set = plasDataset(torch.Tensor(time_series_data).float(), torch.Tensor(classifications).float())
    data_loader = torch.utils.data.DataLoader(training_set, batch_size=500, shuffle=True, num_workers = 4)
    num_epochs = 50

    for epoch in range(num_epochs):
        for n_batch, batch in enumerate(data_loader, 0):
            try:
                data_batch, target_batch = batch
                data, targetss = Variable(data_batch), Variable(target_batch)
                if torch.cuda.is_available():
                    data = data.cuda()
                    targetss = targetss.cuda()
                    simp_error = train_simple(optimizer, data, targetss)
                    print('Accuracy: ' + str(accuracy(targetss, convolutionalNet(data) )))
            except:
                pass
            print('Epoch ' + str(epoch) +': Completed. Loss: ' +str(simp_error.data.cpu().numpy()))
            if epoch+1 % 100 == 0:
                torch.save(convolutionalNet.state_dict(), 'conv_epoch_' + str(epoch))
elif train == 0:
    convolutionalNet.load_state_dict(torch.load('conv_epoch_100', map_location='cpu'))
    print("Network Loaded..")


sample_sub = pd.read_csv('sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])

################# TEST #####
import time

start = time.time()

meta_test = pd.read_csv('test_set_metadata.csv')
chunks = 1000000
for i_c, test in enumerate(pd.read_csv('test_set.csv', chunksize=chunks, iterator=True)):
    print("Data Chunk: " + str(i_c) + " loaded.")
    test['flux_ratio_sq'] = np.power(test['flux'] / test['flux_err'], 2.0)
    test['flux_by_flux_ratio_sq'] = test['flux'] * test['flux_ratio_sq']

    aggs = {
        'mjd': ['min', 'max', 'size'],
        'passband': ['min', 'max', 'mean', 'median', 'std'],
        'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum','skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

    agg_test = test.groupby('object_id').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
        ]
    agg_test.columns = new_columns
    agg_test['mjd_diff'] = agg_test['mjd_max'] - agg_test['mjd_min']
    agg_test['flux_diff'] = agg_test['flux_max'] - agg_test['flux_min']
    agg_test['flux_dif2'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_mean']
    agg_test['flux_w_mean'] = agg_test['flux_by_flux_ratio_sq_sum'] / agg_test['flux_ratio_sq_sum']
    agg_test['flux_dif3'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_w_mean']
    agg_test['flux_deriv_time'] = agg_test['flux_diff'] / agg_test['mjd_diff']
    del agg_test['mjd_max'], agg_test['mjd_min']

    time_series_data_test = []
    rows = [1, 2, 3, 4, 5, 6, 7]
    rows_clipped = [0, 1, 2, 3, 4, 5, 6]
    unique_test_ids = sorted(test['object_id'].unique())
    id_test_group = test.groupby(['object_id'])
    for id in unique_test_ids:
        current_id_group = id_test_group.get_group(id)
        new_id_sample = current_id_group.transpose().as_matrix(columns=current_id_group.transpose().columns[1:])
        new_id_sample = np.array(new_id_sample)
        while new_id_sample.shape[1] < 46:
            new_id_sample = np.insert(new_id_sample, new_id_sample.shape[1], new_id_sample.mean(1), axis=1)
        columns = list(range(0, new_id_sample.shape[1], int(new_id_sample.shape[1]/46)))
        next_sample = new_id_sample[np.array(rows)[:, None], np.array(columns)]
        next_sample_clipped = next_sample[np.array(rows_clipped)[:, None], np.array(list(range(46)))]
        next_sample_clipped = np.array(next_sample_clipped).transpose()
        sample_final = np.expand_dims(next_sample_clipped, axis=0)
        time_series_data_test.append(sample_final)

    time_series_data_test = np.array(time_series_data_test)
    print(time_series_data_test.shape)
    time_series_data_test = torch.Tensor(time_series_data_test)
    time_series_data_test = time_series_data_test
    preds = convolutionalNet(Variable(time_series_data_test))

    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    print(preds.detach().numpy().shape)
    preds = preds.detach().numpy()
    #preds_99 = np.ones(preds.shape[0])
    preds_99 = abs((1 - np.sum(preds[:, 0 ,:, 0], axis=1)))

    # Store predictions
    preds = preds[:,0,:,0]
    preds = np.round(preds)
    preds_df = pd.DataFrame(preds, columns=class_names)
    preds_df['object_id'] = sorted(test['object_id'].unique())
    preds_df['class_99'] = np.round(np.array(preds_99))

    if i_c == 0:
        preds_df.to_csv('predictions.csv',  header=True, mode='a', index=False)
    else:
        preds_df.to_csv('predictions.csv',  header=False, mode='a', index=False)

    del agg_test, preds_df, preds

    if (i_c + 1) % 10 == 0:
        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
