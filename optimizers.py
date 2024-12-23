import numpy as np
from Tensor import Tensor

class optimizers:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None  
    
    def step(self):
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                grad = param.grad if isinstance(param.grad, np.ndarray) else np.array(param.grad)
                param.data -= self.lr * grad

# w= Tensor(np.random.randn(2,2), requires_grad =True)
# b = Tensor(np.random.randn(1,2), requires_grad = True)

# w.grad = np.array([[0.1, -0.2],[0.3, -0.4]])
# b.grad = np.array([0.5, -0.6])

# optimizer = optimizers([w,b], lr =0.01)

# print("Before step")
# print(f"w:{w.data}")
# print(f"w:{w.data}")
# print(f"b:{b.data}")

# optimizer.step()
# print("After step:")
# print(f"w:{w.data}")
# print(f"b:{b.data}")