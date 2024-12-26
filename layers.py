from Tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0/in_features)
        self.weights = Tensor(np.random.randn(in_features, out_features)*scale, requires_grad = True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad = True)
        
    def forward(self,x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        y=x@self.weights+self.bias
        return y
    
    # def parameters(self):
    #     p = self.weights
    #     p = p+self.bias
    #     print(p.requires_grad)
    #     return p
        
    def parameters(self):
        return [self.weights, self.bias]
    
    def __call__(self, x):
        return self.forward(x)
    
# x = Tensor(np.random.randn(5,3), requires_grad = True)
# linear_layer = Linear(3,2)
# output = linear_layer(x)
# print(f"Output :{output}")
# params = linear_layer.parameters()
# print(f"Weights :{params[0].data}")
# print(f"Bias:{params[1].data}")