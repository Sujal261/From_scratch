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
        y_pred = y_pred if isinstance(y_pred, Tensor ) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        epsilon = 1e-12
        y_pred_clamped = np.clip(y_pred.data, epsilon, 1- epsilon)
        BinaryCrossEntropyLoss_loss = -np.mean(y_true.data*np.log(y_pred_clamped)+(1-y_true.data)*np.log(1-y_pred_clamped))
        y_pred = y_pred if isinstance(self, Tensor ) else Tensor(y_pred)
        y_true = y_true if isinstance(self, Tensor) else Tensor(y_true)
        BinaryCrossEntropyLoss_loss = -np.mean(y_true.data*np.log(y_pred.data)+(1-y_true.data)*np.log(1-y_pred.data))
        loss_tensor = Tensor(BinaryCrossEntropyLoss_loss)
        loss_tensor._y_pred = y_pred
        loss_tensor._y_true = y_true
        return loss_tensor

    
    def backward(self, loss_tensor):
        epsilon = 1e-12
        y_pred = loss_tensor._y_pred
        y_pred_clamped = np.clip(loss_tensor._y_pred.data,epsilon, 1- epsilon)
        grad = (y_pred_clamped-loss_tensor._y_true.data)/(y_pred_clamped *(1-y_pred_clamped ))
        y_pred.backward(grad)
    
# y_pred = Tensor([0.9,0.2,0.8],requires_grad=True)
# y_true = Tensor([1,0,1], requires_grad=True)
# bce_loss = BinaryCrossEntropyLoss()
# loss_tensor = bce_loss(y_pred, y_true)
# print("Loss value", loss_tensor.data)
# bce_loss.backward(loss_tensor)
# print("Gradients :", y_pred.grad)
