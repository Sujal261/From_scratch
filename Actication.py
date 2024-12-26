from Tensor import Tensor
import numpy as np

class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, x):
        self.input = x
        # Apply Sigmoid activation
        sigmoid_output = 1 / (1 + np.exp(-x.data))
        self.output = sigmoid_output
        
        # If the tensor requires gradients, return a Tensor with gradient computation function
        requires_grad = x.requires_grad
        
        def backward(grad):
            if self.input.requires_grad:
                # Calculate the gradient for Sigmoid
                sigmoid_grad = grad.data * self.output * (1 - self.output)
                self.input.grad = Tensor(sigmoid_grad)
        
        return Tensor(sigmoid_output, requires_grad=requires_grad, grad_fn=backward if requires_grad else None)


class ReLU:
    def __init__(self):
        self.input = None

    def __call__(self, x):
        self.input = x
        # Apply ReLU activation
        return Tensor(np.maximum(0, x.data), requires_grad=True)

    def backward(self, grad):
        # Calculate the gradient for ReLU
        relu_grad = grad.data * (self.input.data > 0)
        self.input.grad = Tensor(relu_grad)

        
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