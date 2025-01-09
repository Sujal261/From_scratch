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
        


        
# class Tanh:
#     def __call__(self, x):
#         self.x = x
#         self.tanh_output = np.tanh(x.data)
#         return Tensor(self.tanh_output, requires_grad=True)
    
#     def grad_fn(self):
#         tanh_grad = 1- self.tanh_output**2
#         self.x.grad = tanh_grad
        
# x = Tensor(np.array([0.5, -0.5, 1.0]), requires_grad = True)
# sigmoid = Sigmoid()
# y_sigmoid = sigmoid(x)
# print(f"Sigmoid output :{y_sigmoid.data}")
# sigmoid.backward()
# print(f"Sigmoid gradiient :{x.grad}")

# def test_gradient_flow():
#     # Create input tensor
#     x = Tensor(np.array([0.5, -0.5, 1.0]), requires_grad=True)
    
#     # Test Sigmoid
#     print("Testing Sigmoid:")
#     sigmoid = Sigmoid()
#     y = sigmoid(x)
#     print(f"Forward pass output: {y.data}")
#     y.backward(np.ones_like(y.data))
#     print(f"Gradient at x: {x.grad}\n")
    
#     # Reset gradient
#     x.grad = None
    
#     # Test ReLU
#     print("Testing ReLU:")
#     relu = ReLU()
#     y = relu(x)
#     print(f"Forward pass output: {y.data}")
#     y.backward(np.ones_like(y.data))
#     print(f"Gradient at x: {x.grad}")

# test_gradient_flow()