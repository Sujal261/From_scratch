import numpy as np 

class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None 
        self._backward_fns = []
        self.grad_fn = grad_fn 
        
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        
        # If no gradient provided, use ones with same shape as data
        if grad is None:
            grad = np.ones_like(self.data)
        
        # Ensure gradient shape matches data shape
        if grad.shape != self.data.shape:
            grad = np.mean(grad, axis=0, keepdims=True)
        
        # Accumulate gradient
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
            
        # Execute backward functions in reverse order
        for backward_fn in reversed(self._backward_fns):
            backward_fn(grad)
            
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        def backward_fn(grad):
            if self.requires_grad:
                self.backward(grad)
            if other.requires_grad:
                other.backward(grad)
        
        result = Tensor(data, requires_grad)
        if requires_grad:
            result._backward_fns.append(backward_fn)
        return result
                
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data - other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        def backward_fn(grad):
            if self.requires_grad:
                self.backward(grad)
            if other.requires_grad:
                other.backward(-grad)
        
        result = Tensor(data, requires_grad)
        if requires_grad:
            result._backward_fns.append(backward_fn)
        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        def backward_fn(grad):
            if self.requires_grad:
                self.backward(grad * other.data)
            if other.requires_grad:
                other.backward(grad * self.data)
        
        result = Tensor(data, requires_grad)
        if requires_grad:
            result._backward_fns.append(backward_fn)
        return result
                
    def __matmul__(self, other):
        data = np.dot(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        
        def backward_fn(grad):
            if self.requires_grad:
                # Ensure correct gradient shape for matrix multiplication
                self.backward(np.dot(grad, other.data.T))
            if other.requires_grad:
                other.backward(np.dot(self.data.T, grad))
        
        result = Tensor(data, requires_grad)
        if requires_grad:
            result._backward_fns.append(backward_fn)
        return result
    
    def __pow__(self, power):
        data = self.data**power
        requires_grad = self.requires_grad
        
        def backward_fn(grad):
            if self.requires_grad:
                self.backward(grad * power * self.data**(power - 1))
        
        result = Tensor(data, requires_grad)
        if requires_grad:
            result._backward_fns.append(backward_fn)
        return result
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad
        
        def backward_fn(grad):
            if self.requires_grad:
                self.backward(grad / other.data)
            if other.requires_grad:
                other.backward(-grad * self.data / (other.data**2))
        
        result = Tensor(data, requires_grad)
        if requires_grad:
            result._backward_fns.append(backward_fn)
        return result
                
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"

class Linear:
    def __init__(self, in_features, out_features):
        # Proper weight initialization with Xavier/Glorot initialization
        self.weights = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(1.0 / in_features), 
            requires_grad=True
        )
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)
        
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x @ self.weights + self.bias
    
    def parameters(self):
        return [self.weights, self.bias]
    
    def __call__(self, x):
        return self.forward(x)

class MSELoss:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        
        # Compute difference
        diff = y_pred - y_true
        
        # Compute mean squared error
        mse_loss = np.mean(diff.data**2)
        
        # Create a tensor with requires_grad
        loss_tensor = Tensor(mse_loss, requires_grad=True)
        
        # Define backward function for loss
        def backward_fn(grad=None):
            # Compute gradient of loss with respect to predictions
            # Average gradient if shapes don't match
            loss_grad = 2 * (y_pred.data - y_true.data) / np.prod(y_true.data.shape)
            
            # Backpropagate the gradient
            y_pred.backward(loss_grad)
        
        # Add backward function to the loss tensor
        loss_tensor._backward_fns.append(backward_fn)
        
        return loss_tensor

class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for param in self.params:
            if param.requires_grad:
                param.grad = None
    
    def step(self):
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                # Ensure gradient shape matches parameter shape
                grad = param.grad
                if grad.shape != param.data.shape:
                    # Average or reshape gradient if needed
                    grad = np.mean(grad, axis=0, keepdims=True)
                
                # Update parameter data
                param.data -= self.lr * grad

# Training loop
def train():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data
    x = np.random.randn(100, 1)
    y = 2*x + 1
    train_split = int(0.8*len(x))
    x_train, y_train = x[:train_split], y[:train_split]

    # Convert to tensors with requires_grad
    x_train_tensor = Tensor(x_train)
    y_train_tensor = Tensor(y_train)

    # Initialize model, loss, and optimizer
    model = Linear(1, 1)
    loss_fn = MSELoss()
    optimizer = Optimizer(params=model.parameters(), lr=0.01)

    # Training loop
    epochs = 300
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.forward(x_train_tensor)
        
        # Compute loss
        loss = loss_fn(y_pred, y_train_tensor)
        
        # Zero out previous gradients
        optimizer.zero_grad()
        
        # Backward pass (compute gradients)
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print loss periodically
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data}")

# Run training
train()