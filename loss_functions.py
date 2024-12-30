from Tensor import Tensor
import numpy as np
class MSELoss:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        diff = y_pred.data - y_true.data
        mse_loss = np.mean(diff**2)
        requires_grad = y_pred.requires_grad or y_true.requires_grad
        
        def grad_fn(self):
           grad = 2 * (y_pred.data - y_true.data) / len(y_true.data)
           y_pred.backward(grad)
           
        loss_tensor = Tensor(mse_loss, requires_grad=requires_grad, grad_fn=grad_fn if requires_grad else None)
        return loss_tensor 

class BinaryCrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor ) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        epsilon = 1e-12
        y_pred_clamped = np.clip(y_pred.data, epsilon, 1- epsilon)
        Loss= -np.mean(y_true.data*np.log(y_pred_clamped)+(1-y_true.data)*np.log(1-y_pred_clamped))
        requires_grad = y_pred.requires_grad or y_true.requires_grad
    
        def grad_fn(self):
             
             if self.y_pred.requires_grad:
               epsilon = 1e-12
               y_pred_clamped = np.clip(y_pred.data,epsilon, 1- epsilon)
               grad = ((y_pred_clamped-y_true.data)/(y_pred_clamped *(1-y_pred_clamped )))/len(y_true.data)
               y_pred.backward(grad)
        loss_tensor = Tensor(Loss, requires_grad= requires_grad, grad_fn=grad_fn if  requires_grad else None)
        return loss_tensor
    
# y_pred = Tensor([0.9,0.2,0.8],requires_grad=True)
# y_true = Tensor([1,0,1], requires_grad=True)
# bce_loss = BinaryCrossEntropyLoss()
# loss_tensor = bce_loss(y_pred, y_true)
# print("Loss value", loss_tensor.data)
# bce_loss.backward(loss_tensor)
# print("Gradients :", y_pred.grad)
