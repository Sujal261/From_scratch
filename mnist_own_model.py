import numpy as np  
import matplotlib.pyplot as plt 
from Tensor import Tensor
from optimizers import Optimizer
from layers import Linear, Flatten, MaxPool, Conv2D
from Actication import ReLU, Softmax
from loss_functions import CategoricalCrossEntropyLoss
from sklearn.datasets import load_digits
np.random.seed(42)
#loading the mnist data
mnist = load_digits()
data = mnist['data']  
target = mnist['target']
train_split = int(0.8 * len(data))

#Loading train and test splitd
train_data1 = data[:train_split, :]
train_target1 = target[:train_split]
test_data1 = data[train_split:,:]
test_target = target[train_split:]

#Normalizing the data 
train_data2 = train_data1/255
test_data2 = test_data1/255

train_data3 = train_data2.reshape(-1,1,8,8)
test_data = train_data2.reshape(-1,1,8,8)


target, image = train_target1[2],train_data3[2].squeeze()
plt.imshow(image, cmap='gray')
plt.title(target)
plt.show()

indices = np.random.permutation(len(train_data3))  # Get shuffled indices

# Apply the shuffle to both data and target
train_data = train_data3[indices]
train_target= train_target1[indices]




class MNIST:
    def __init__(self, in_shape, out_shape, hidden_units):
        self.layer1 = Conv2D(in_features=in_shape, out_features=hidden_units, stride=1, padding=1, kernel_size=(2,2))
        self.layer2 = ReLU()
        self.layer3 = Conv2D(in_features=hidden_units, out_features=hidden_units,stride =1 , padding = 1, kernel_size=(2,2) )
        self.layer4 = ReLU()
        self.layer5 = MaxPool(stride = 1, kernel_size = (2,2))
        self.layer6 = Flatten()
        self.layer7 = Linear(in_features=hidden_units*81, out_features=out_shape)
        self.layer8 = Softmax()
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x
    
    def params(self):
        return self.layer1.parameters()+self.layer3.parameters()+self.layer7.parameters()
    
model = MNIST(in_shape=1, out_shape=10, hidden_units=16)

loss_fn = CategoricalCrossEntropyLoss()
optimizer = Optimizer(model.params(), lr = 0.1)

num_epochs = 10
batch_size= 32
num_batches = len(train_data)//batch_size

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx in range(num_batches):
        batch_start = batch_idx *batch_size
        batch_end = batch_start+batch_size
        batch_data = train_data[batch_start:batch_end]
        batch_target = train_target[batch_start:batch_end]
        
        batch_data_tensor = Tensor(batch_data, requires_grad=True)
        batch_target_tensor = Tensor(batch_target, requires_grad=True)
        
        prediction = model.forward(batch_data_tensor)
        
        loss = loss_fn(prediction, batch_target_tensor)
        total_loss+=loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    avg_loss = total_loss/num_batches
    print(f'Loss at  {epoch+1} is {avg_loss}')


correct = 0
total = len(test_data)
num_test_batches = total // batch_size

for batch_idx in range(num_test_batches):
    batch_start = batch_idx * batch_size
    batch_end = min(batch_start + batch_size ,len(train_data))
    batch_data = test_data[batch_start:batch_end]
    batch_target = test_target[batch_start:batch_end]

   
    batch_data_tensor = Tensor(batch_data, requires_grad=True)

   
    predictions = model.forward(batch_data_tensor)

    
    predicted_classes = np.argmax(predictions.data, axis=1)
    correct += np.sum(predicted_classes == batch_target)


accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
        
        
        



# target, image = train_target[2],train_data[2].squeeze()
# plt.imshow(image, cmap='gray')
# plt.title(target)
# plt.show()