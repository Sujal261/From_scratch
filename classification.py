import numpy as np 
import matplotlib.pyplot as plt
from loss_functions import MSELoss, BinaryCrossEntropyLoss
from optimizers import Optimizer
from layers import Linear 
from Tensor import Tensor
from Actication import ReLU, Sigmoid
from plotting_curves import loss_curve
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=100, noise = 0.1, factor = 0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# def plot(data, labels):
#     plt.scatter(X[:,0],X[:,1], c= y, cmap = 'viridis')
#     plt.show()
    
# plot(X, y)
X_train_tensor = Tensor(X_train, requires_grad=True)
y_train_tensor = Tensor(y_train, requires_grad=True)
    
class classification:
    def __init__(self, in_features, out_features):
        self.linear_layer1 = Linear(in_features,8)
        self.relu = ReLU()
        self.linear_layer2 = Linear(8,out_features)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        
        # Debug prints
        x1 = self.linear_layer1(x)
        print(f"After linear1 - mean: {x1.data.mean():.4f}, std: {x1.data.std():.4f}")
        
        x2 = self.relu(x1)
        print(f"After ReLU - mean: {x2.data.mean():.4f}, std: {x2.data.std():.4f}")
        
        x3 = self.linear_layer2(x2)
        print(f"After linear2 - mean: {x3.data.mean():.4f}, std: {x3.data.std():.4f}")
        
        x4 = self.sigmoid(x3)
        print(f"After sigmoid - mean: {x4.data.mean():.4f}, std: {x4.data.std():.4f}")
        
        return x4
    def parameters(self):

        return self.linear_layer1.parameters()+self.linear_layer2.parameters()
        
model =classification(2,1)
loss_fn = BinaryCrossEntropyLoss()
# print(model.parameters())
optimizer = Optimizer(params = model.parameters(), lr = 0.01)

#Making a training loop 
epochs = 100
for epoch in range(epochs):
    y_pred = model.forward(X_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    for i, param in enumerate(model.parameters()):
      if param.grad is not None:
        print(f"Parameter {i}: Gradient - Mean: {np.mean(param.grad):.4f}, Std: {np.std(param.grad):.4f}")
      else:
        print(f"Parameter {i}: Gradient is None")

    for i, param in enumerate(model.parameters()):
       print(f"Before step - Parameter {i}: Mean: {np.mean(param.data):.4f}, Std: {np.std(param.data):.4f}")

    optimizer.step()

    for i, param in enumerate(model.parameters()):
       print(f"After step - Parameter {i}: Mean: {np.mean(param.data):.4f}, Std: {np.std(param.data):.4f}")

    
    if epoch %10 ==0:
         print(f"Epochs:{epoch} train loss:{loss.data}")