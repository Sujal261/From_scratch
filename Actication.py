from Tensor import Tensor
import numpy as np

class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, x):
        self.input = x
       
        x_data = np.clip(x.data, -500, 500)  
        self.output = 1 / (1 + np.exp(-x_data))
        requires_grad = x.requires_grad
        
        def backward(grad):
            if self.input.requires_grad:
                sigmoid_grad = grad * self.output * (1 - self.output)
                self.input.backward(sigmoid_grad)
        
        return Tensor(self.output, requires_grad=requires_grad, grad_fn=backward if requires_grad else None)


class ReLU:
    def __init__(self):
        self.input = None
        self.output= None

    def __call__(self, x):
        self.input = x
        self.output = np.maximum(0, x.data)
        output = self.output
        requires_grad = x.requires_grad

        def grad_fn(grad):
            if self.input.requires_grad:
              relu_grad =  grad*(self.input.data > 0)
              self.input.backward(relu_grad)
           
        return Tensor(output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
    
class Softmax:
    def __init__(self):
        self.input = None 
        self.output = None 
        
    def __call__(self, x):
        self.input = x
        x_data = x.data - np.max(x.data, axis = 1, keepdims=True)
        self.output = np.exp(x_data)/np.sum(np.exp(x_data, axis = 1, keepdims = True))
        requires_grad = x.requires_grad
        
        def grad_fn(grad):
           if self.input.requires_grad:
               s= self.output
               c = len(s)
               softmax_grad = np.zeros_like(s)
               
               for i  in range(c):
                   for j in range(c):
                       if i == j:
                           softmax_grad[i]+=softmax_grad[j]*s[i]*(1-s[i])
                       else:
                           softmax_grad[i]+=softmax_grad[j]*(-s[i]*s[j])
               self.input.backward(softmax_grad)
                
                           
        return Tensor(self.output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
        


        
