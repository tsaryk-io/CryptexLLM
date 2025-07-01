import numpy as np
import torch
from torch import nn
import numpy as np

def get_loss_function(loss_name):
    # Returns an instance of the specified loss function.
    loss_name = loss_name.upper()
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAE':
        return nn.L1Loss()
    elif loss_name == 'MAPE':
        return MAPELoss()
    elif loss_name == 'MADL':
        return MADLLoss()
    elif loss_name == 'GMADL':
        return GMADLLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_name}")

def get_metric_function(metric_name):
    # Returns an instance of the specified evaluation metric.
    metric_name = metric_name.upper()
    if metric_name == 'MSE':
        return nn.MSELoss()
    elif metric_name == 'MAE':
        return nn.L1Loss()
    elif metric_name == 'MAPE':
        return MAPELoss()
    elif metric_name == 'MDA':
        return MDAMetric()
    elif metric_name == 'SHARPE':
        return SharpeRatioMetric()
    else:
        raise ValueError(f"Unsupported metric type: {metric_name}")


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error:
        MAPE = mean( |[pred - true] / true| )
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps # to avoid division by zero

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # ensure true is nonzero
        denom = torch.where(torch.abs(true) < self.eps,
                            torch.full_like(true, self.eps),
                            true)
        return torch.mean(torch.abs((pred - true) / denom))


class MDAMetric(nn.Module):
    """
    Mean Directional Accuracy (MDA)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        # pred, true shape: [batch, seq_len, feature_dim] or [batch, seq_len]
        # Compare change direction relative to previous timestep

        if pred.shape[1] < 2:
            print(f"[Warning] Not enough steps to compute MDA. pred.shape: {pred.shape}")
            return torch.tensor(0.0, device=pred.device)

        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]

        correct = (pred_diff * true_diff) > 0  # boolean tensor: True if same direction
        mda = correct.float().mean()  # take mean accuracy over all elements

        return mda

class SharpeRatioMetric(nn.Module):
    """
    Computes the Sharpe Ratio for a batch of returns (predictions).
    Assumes risk-free rate = 0.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # to avoid division by zero

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true shape: [batch, seq_len, feature_dim] or [batch, seq_len]
        Computes over the prediction period only.
        """
        # calculate returns as diff relative to previous timestep
        returns = pred[:, 1:] - pred[:, :-1]

        mean_return = returns.mean()
        std_return = returns.std()

        # prevent divide-by-zero
        sharpe_ratio = mean_return / (std_return + self.eps)

        return sharpe_ratio

class MADLLoss(nn.Module):
    # Mean Absolute Directional Loss (MADL) by F Michankow (2023)
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true: [batch, seq_len] or [batch, seq_len, 1] (predicted and true returns)
        """
        # Ensure same shape
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")

        product_sign = torch.sign(true * pred)  # sign(Ri * R̂i)
        abs_return = torch.abs(true)

        loss = (-1.0) * product_sign * abs_return

        return loss.mean()

class GMADLLoss(nn.Module):
    # Generalized Mean Absolute Directional Loss (GMADL) by F. Michankow (2024)
    def __init__(self, a=1.0, b=1.0):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        pred, true: [batch, seq_len] or [batch, seq_len, 1] (predicted and true returns)
        """
        # ensure same shape
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")

        # The paper uses a=1000 and b=1:5
        product = self.a * true * pred  # element-wise Ri * R̂i

        sigmoid_term = 1.0 / (1.0 + torch.exp(-product))  # 1 / (1 + exp(-a Ri R̂i))

        adjustment = sigmoid_term - 0.5  # ( ... ) - 0.5

        weighted_abs_return = torch.abs(true) ** self.b  # |Ri|^b

        loss = -1.0 * adjustment * weighted_abs_return

        # Mean over all elements
        return loss.mean()


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
