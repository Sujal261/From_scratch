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

def plot(data, labels):
    plt.scatter(X[:,0],X[:,1], c= y, cmap = 'viridis')
    plt.show()
    
plot(X, y)
    
class classification:
    def __init__(self, in_features, out_features):
        self.linear_layer1 = Linear(in_features,8)
        self.relu = ReLU()
        self.linear_layer2 = Linear(8,out_features)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        x= self.linear1(x)
        x= self.relu(x)
        x=self.linear_layer2(x)
        x=self.sigmoid(x)
        return x
model =classification()
loss_fn = BinaryCrossEntropyLoss()
optimizer = Optimizer(params = model.parameters(), lr = 0.01)
