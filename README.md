# EE399-A-HW6
## Performance
Marcus Chen, May 19, 2023

In previous projects, we developed machine learning models, specifically neural networks catered toward specific datasets, but a neural network model should be able to work for any piece of information, granted they are similar to the dataset. In this project, we will be using the Shallow Recurrent Decoder (SHRED) neural network from:

https://github.com/shervinsahba/pyshred

to train and predict values of sea surface temperatures (SST). We will then analyze the performance of SHRED when we modify the data based on time lag, Gaussian noise, and number of sensors. 

### Introduction:
In previous projects, we trained and optimized a model – a polynomial function, an LDA, an SVD, a decision tree, or now a neural network – to fit around one dataset. But when we apply a machine learning model for the usage of an AI system or algorithm, we’re often not confronting information from the comfort of a dataset; in a number detector that is trained on MNIST, the input will more-than-likely be unique data when it is past the testing phase. The same can be said with any model.

In this project, we will use the SHRED neural network on their sample SST data. An example code is provided for how to train, test, and display the results of the SHRED network. 

The MSE error and the reconstruction should appear like this:

```
0.019834021
```

![download](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/960ddbaf-dbbc-4900-af04-5b26be000494)

![download (1)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/05c8b176-2162-4ba2-a49d-8b581a874e02)

The shape of the dataset and the modification of SHRED appear among two variables: the time lag and the number of sensors. The time lag, measured in weeks, is a variable used to measure the trajectory length of the data, with the default of 52 corresponding to exactly one year of measurements. The number of sensors determines the size of the input of the SHRED model used to then perform data regression. 

In the first part of this project, we will first check the performance of the network as a function of the time lag.

In the second part of the project, we will check the performance of the network as a function of Gaussian noise, done by adding Gaussian noise of various intensity to the data.

In the last part of the project, we will check the performance of the network as a function of the number of sensors. 

### Theoretical Background:
#### Shallow Recurrent Decoder (SHRED):
A merge between an LSTM and a shallow decoder network (SDN), used to reconstruct high-dimensional spatio-temporal fields from the trajectory of sensor measurements. The formal architecture can be written as:

```math
\mathcal{H}( \{ y_i \} _{i=t+k} ^{t}) = \mathcal{F}( \mathcal{G}( \{ y_i \} _{i=t+k} ^{t}) ;W_{LSTM}) ;W_{FFNN})
```

With $\mathcal{F}$ referring to a FFNN and $\mathcal{G}$ referring to an LSTM. 

From the paper: “Sensing with shallow recurrent decoder networks” by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz. 

#### Gaussian Noise:
A signal noise that has a probability density function (PDF) similar to a normal or Gaussian distribution. The probability density of a random variable $z$ is given by:

```math
p_G(z) = \frac{1}{\sigma \sqrt{2 \pi}}e^{\frac{(z-\mu)^2}{2\sigma^2}}
```
where $\mu$ is the mean and $\sigma$ is standard deviation.

### Algorithm and Interpretation:
Following the example code as guidance, the most important step is to first load the data using load_data from the Github’s processdata.py and set the constants used as control for the various analyses:
```
import numpy as np
from processdata import load_data
from scipy.io import loadmat

num_sensors = 3 
lags = 52
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
```

Out of the number of columns, random ones are chosen to be the location for the set scalar of sensors:
```
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```

1,000 indexes are randomly decided for training, testing and validation data subsets, based on the size of the row space minus the lag:
```
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```

In order to scale the data, sklearn’s MinMaxScaler() is used to pre-process the data into a more manageable numerical range, where data can then be then organized into training, validation, and testing input and output:
```
from sklearn.preprocessing import MinMaxScaler
import torch

sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)
```

The data can then be organized using the Github’s TimeSeriesDataset from processdata.py:
```
### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```

To make the SHRED model and to fit the model with the training data, use the Github's SHRED and fit methods from models.py:
```
import models

shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
```

To make a reconstruction of any of the data, use the TimeSeriesDataset model from before and perform an inverse transformation on the datasets:
```
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
```

The MSE error is based on:
```
error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
```

To graph the data, use the Github’s load_full_SST() method from processdata.py and create a set for the truth and reconstruction based on the selected indices:
```
full_test_truth = full_SST[test_indices, :]

# replacing SST data with our reconstruction
full_test_recon = full_test_truth.copy()
full_test_recon[:,sst_locs] = test_recons
```

Reshape and graph using matplotlib:
```
# reshaping to 2d frames
for x in [full_test_truth, full_test_recon]:
    x.resize(len(x),180,360)

plotdata = [full_test_truth, full_test_recon]
labels = ['test truth','test recon']
fig, ax = plt.subplots(1,2,constrained_layout=True,sharey=True)
for axis,p,label in zip(ax, plotdata, labels):
    axis.imshow(p[0])
    axis.set_aspect('equal')
    axis.text(0.1,0.1,label,color='w',transform=axis.transAxes)
```

To evaluate the functions as various functions, an array of varying values are created as well as an array for MSE errors:
```
n_stuff = np.arange(1,21)
error = np.zeros(20)
```

The values are then used to build a loop that continually selects data, trains, and tests the data:
```
for j in range(len(n_stuff)):
  
  *Machine learning processes over 100 epochs, with constant replaced with n_stuff at index j*

  FST.append(full_test_truth)
  FSR.append(full_test_recon)
print(error)
```

In the case of evaluating Gaussian noise, a list of Gaussian intensity values are determined and they are multiplied to an array made by np.random.normal() which shares the same shape to the loading data:
```
n_noise = np.arange(0,20) / 10
```
```
noise = n_noise[j] * np.random.normal(size=load_X.shape)
...
transformed_X = sc.transform(load_X + noise)
```

### Results:
In order to analyze SHRED as a function of time lag, a series of 20 values were created incremented by 5, giving a range of 5-100. The results are:

![download (2)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/ff46c5f9-8789-4d83-9084-855d6606b882)

Where the reconstruction with the most amount of error looks like:
![download (3)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/48bde81b-7e57-42db-9a1e-520d8a9aaf1e)

And the reconstruction with the least amount of error looks like:
![download (4)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/dc224140-6384-4884-b900-43b52ae751d0)

There is a general trend that the more time lag there is, the better the model was at performing a regression, though not by much. 

In order to analyze SHRED as a function of Gaussian noise, a series of 20 noise intensity values from 0 to 1.9 were created. The results are:

![download (5)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/7f76ea46-9507-42b5-93f7-1e8afec5f22a)

And the reconstruction with the most amount of error looks like:
![download (7)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/dd600207-de4f-446d-b140-31e02ade16d7)

Where the reconstruction with the least amount of error (no noise) looks like:
![download (6)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/bc4fe255-1331-4722-be05-0d799966b180)


Noise seems to have a pretty profound effect on the MSE error of the SHRED network, but it does not go above 0.25.

In order to analyze SHRED as a function of the number of sensors, a series of 20 values from 1 to 20 were created. The results are:

![download (8)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/c04e1257-1fb7-48a8-9821-bbd522ccb470)

Where the reconstruction with the most amount of error looks like:
![download (9)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/929aef7b-9079-4d7b-8498-5d7e23d1592a)

And the reconstruction with the least amount of error looks like:
![download (10)](https://github.com/marcuschen001/EE399-A-HW6/assets/66970342/a88159fe-b6a0-4607-bad3-a38071923097)

Similar to the time lag, there seems to be a negative correlation between the number of sensors and the MSE error. Though in the case of sensors, there is a big difference between 1 sensor and more. 

### Summary and Conclusion:
Over the course of this project, we saw the effect of modifying constants and how it affected the performance of the SHRED network overall. While some changes made very minute effects on the overall performance, some – like the increased intensity in Gaussian noise – were pretty profound. 

In the beginning of the class, overfitting was mentioned to be a concern when it came to modular development, and even here, it is affirmed once again. When considering the design of machine learning models, it is important to think beyond only the training and testing data.
