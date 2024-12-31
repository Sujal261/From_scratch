from Tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0/in_features)
        self.weights = Tensor(np.random.randn(in_features, out_features)*scale, requires_grad = True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad = True)
        
    def __call__(self,x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.x = x
        output=x.data@self.weights.data+self.bias.data
        requires_grad = self.weights.requires_grad or self.bias.requires_grad or x.requires_grad
        
        def grad_fn(grad):
            if self.weights.requires_grad:
                self.weights.grad = x.data.T @ grad
            if self.bias.requires_grad:
                self.bias.grad = np.sum(grad, axis = 0, keepdims=True)
            if x.requires_grad:
                x.backward(grad @ self.weights.data.T)
        return Tensor(output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
   
        
    def parameters(self):
        return [self.weights, self.bias]
    
    
# x = Tensor(np.random.randn(5,3), requires_grad = True)
# linear_layer = Linear(3,2)
# output = linear_layer(x)
# print(f"Output :{output}")
# params = linear_layer.parameters()
# print(f"Weights :{params[0].data}")
# print(f"Bias:{params[1].data}")