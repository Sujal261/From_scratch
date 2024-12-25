from Tensor import Tensor
import numpy as np

class Sigmoid:
    def __call__(self, x):
        self.x = x 
        self.sigmoid_output = 1/(1+np.exp(-x.data))
        return Tensor(self.sigmoid_output, requires_grad=True)
    
    def grad_fn(self):
        sigmoid_grad = self.sigmoid_output * (1- self.sigmoid_output)
        self.x.grad = sigmoid_grad
        
class ReLU:
    def __call__(self, x):
        self.x = x 
        self.relu_output = np.maximum(0, x.data)
        return Tensor(self.relu_output, requires_grad=True)
    def grad_fn(self):
        relu_grad = np.where(self.relu_output>0, 1 , 0)
        self.x.grad = relu_grad
        
class Tanh:
    def __call__(self, x):
        self.x = x
        self.tanh_output = np.tanh(x.data)
        return Tensor(self.tanh_output, requires_grad=True)
    
    def grad_fn(self):
        tanh_grad = 1- self.tanh_output**2
        self.x.grad = tanh_grad
        
# x = Tensor(np.array([0.5, -0.5, 1.0]), requires_grad = True)
# sigmoid = Sigmoid()
# y_sigmoid = sigmoid(x)
# print(f"Sigmoid output :{y_sigmoid.data}")
# sigmoid.backward()
# print(f"Sigmoid gradiient :{x.grad}")