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
        grad = 2 * (loss_tensor._y_pred.data - loss_tensor._y_true.data) / len(loss_tensor._y_true.data)
        loss_tensor._y_pred.backward(grad) # Propagate gradient to y_pred

class BinaryCrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred if isinstance(self, Tensor ) else Tensor(y_pred)
        y_true = y_true if isinstance(self, Tensor) else Tensor(y_true)
        BinaryCrossEntropyLoss_loss = -np.mean(y_true.data*np.log(y_pred.data)+(1-y_true.data)*np.log(1-y_pred.data))
        loss_tensor = Tensor(BinaryCrossEntropyLoss_loss)
        loss_tensor._y_pred = y_pred
        loss_tensor._y_true = y_true
        return loss_tensor
    def backward(self, loss_tensor):
        grad = (loss_tensor._y_pred.data-loss_tensor._y_true.data)/(loss_tensor._y_pred.data*(1-loss_tensor._y_pred.data))
        