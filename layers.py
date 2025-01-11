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
    
    
class Flatten:
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, x):
        self.input = x
        p =x.data
        self.output = p.flatten()
        requires_grad = x.requires_grad
        
        def grad_fn(grad):
            if x.requires_grad:
                flatten_grad = grad.reshape(x.data.shape)
                self.input.backward(flatten_grad)
        return Tensor(self.output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)

    
class MaxPool:
    def __init__(self):
        self.input = None 
        self.output = None
        
    def __call__(self, x, stride, kerne_size ):
        self.input = x
        e, f = kerne_size
        i, j = x.data.shape
        
        g,h = (i-e)//stride+1, (j-f)//stride+1
        m=[]
        max_indices=[]
        for a in range(0,i, stride):
            z=[]
            for b in range(0,j, stride):
                window= x.data[a:a+e, b:b+f]
                
                p=np.max(window)
                
                flat_index= np.argmax(window)
                max_row , max_col = np.unravel_index(flat_index, window.shape)
                m.append(p)
                z.append((a+max_row, b+max_col))
            max_indices.append(z)
            
        self.output = np.array(m).reshape(g, h)
        requires_grad = x.requires_grad
        max_indices = np.array(max_indices)
        
        
        def grad_fn(grad):
            if x.requires_grad:
                maxpool_grad = np.zeros_like(self.input.data)
               
                for a in range(g):
                    for b in range(h):
                        max_row, max_col = max_indices[a][b]
                        maxpool_grad[max_row, max_col]+=grad[a,b]
                        
                self.input.backward(maxpool_grad)
        return Tensor(self.output, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None) 
    
