import numpy as np  
import matplotlib.pyplot as plt 
from Tensor import Tensor
from optimizers import Optimizer
from layers import Linear, Flatten, MaxPool, Conv2D
from sklearn.datasets import load_digits
np.random.seed(42)
#loading the mnist data
mnist = load_digits()
data = mnist['data']  
target = mnist['target']
train_split = int(0.8 * len(data))

#Loading train and test split
train_data1 = data[:train_split, :]
train_target1 = target[:train_split]
test_data1 = data[train_split:,:]
test_target = target[train_split:]

#Normalizing the data 
train_data2 = train_data1/255
test_data2 = test_data1/255

train_data3 = train_data2.reshape(-1,1,8,8)
test_data = train_data2.reshape(-1,1,8,8)

indices = np.random.permutation(len(train_data3))  # Get shuffled indices

# Apply the shuffle to both data and target
train_data = train_data3[indices]
train_target= train_target1[indices]

batch_size = 32






target, image = train_target[2],train_data[2].squeeze()
plt.imshow(image, cmap='gray')
plt.title(target)
plt.show()