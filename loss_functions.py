from Tensor import Tensor
import numpy as np
class MSELoss:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        diff = y_pred - y_true
        mse_loss = np.mean(diff.data**2)
        loss_tensor=Tensor(mse_loss, requires_grad=True)
        loss_tensor._y_pred = y_pred
        loss_tensor._y_true = y_true
        return loss_tensor

    def backward(self, loss_tensor):
        grad = 2 * (loss_tensor._y_pred.data - loss_tensor._y_true.data) / len(loss_tensor._y_true.data)
        loss_tensor._y_pred.backward(grad) # Propagate gradient to y_pred

# y_pred = Tensor(np.array([3.0, 2.5, 1.0]))
# y_true = Tensor(np.array([3.0,3.0,1.0]))
# loss_fn = MSELoss()
# loss = loss_fn(y_pred, y_true)
# print(f"loss :{loss.data}")
# loss_fn.backward()
# print(f"Gradient of y_pred :{y_pred.grad}")