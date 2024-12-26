from Tensor import Tensor
import numpy as np
class MSELoss:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        diff = y_pred - y_true
        mse_loss = np.mean(diff.data**2)
        loss_tensor=Tensor(mse_loss)
        loss_tensor._y_pred = y_pred
        loss_tensor._y_true = y_true
        return loss_tensor

    def backward(self, loss_tensor):
        y_pred = loss_tensor._y_pred
        grad = 2 * (loss_tensor._y_pred.data - loss_tensor._y_true.data) / len(loss_tensor._y_true.data)
        y_pred.backward(grad) # Propagate gradient to y_pred

class BinaryCrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        epsilon = 1e-12
        y_pred_clamped = np.clip(y_pred.data, epsilon, 1 - epsilon)
        loss = -np.mean(y_true.data * np.log(y_pred_clamped) + (1 - y_true.data) * np.log(1 - y_pred_clamped))
        loss_tensor = Tensor(loss, requires_grad=True)  # Set requires_grad=True
        loss_tensor._y_pred = y_pred
        loss_tensor._y_true = y_true
        return loss_tensor

    def backward(self, loss_tensor):
        epsilon = 1e-12
        y_pred = loss_tensor._y_pred
        y_true = loss_tensor._y_true
        y_pred_clamped = np.clip(y_pred.data, epsilon, 1 - epsilon)
        
        # Calculate the gradient of the loss with respect to y_pred
        grad = (y_pred_clamped - y_true.data) / (y_pred_clamped * (1 - y_pred_clamped))
        
        # Backpropagate the gradient to y_pred
        y_pred.backward(grad)

    
# Example test case
y_pred = Tensor(np.array([[0.8], [0.1], [0.5]]), requires_grad=True)  # Predicted probabilities
y_true = Tensor(np.array([[1], [0], [1]]), requires_grad=True)        # True labels

loss_fn = BinaryCrossEntropyLoss()
loss = loss_fn(y_pred, y_true)
print(f"Loss: {loss.data}")  # Should compute the loss value

loss_fn.backward(loss)  # Backpropagate
print(f"Gradient: {y_pred.grad}")  # Should show valid gradients
