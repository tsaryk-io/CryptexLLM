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
    elif loss_name == 'DLF':
        return DLFLoss()
    elif loss_name == 'ASYMMETRIC':
        return AsymmetricLoss()
    elif loss_name == 'QUANTILE':
        return QuantileLoss()
    elif loss_name == 'SHARPE_LOSS':
        return SharpeRatioLoss()
    elif loss_name == 'TRADING_LOSS':
        return TradingLoss()
    elif loss_name == 'ROBUST':
        return RobustLoss()
    elif loss_name in ['COMPREHENSIVE', 'TRADING_FOCUSED', 'DIRECTIONAL_FOCUSED', 'BASIC']:
        # Import adaptive loss here to avoid circular imports
        from utils.adaptive_loss import create_adaptive_loss
        return create_adaptive_loss(combination=loss_name.lower())
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

class DLFLoss(nn.Module):
    # Directional Loss Function:
    def __init__(self, lambda_weight=0.5):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.base_loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
            true = true.squeeze(-1)

        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]
        direction_mismatch = (torch.sign(pred_diff) != torch.sign(true_diff)).float()
        direction_loss = direction_mismatch.mean()

        base = self.base_loss(pred, true)

        total_loss = self.lambda_weight * direction_loss + (1 - self.lambda_weight) * base
        return total_loss


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss function that penalizes underestimation and overestimation differently.
    Useful for trading scenarios where false signals have different costs.
    """
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for underestimation (missing upward moves)
        self.beta = beta    # Weight for overestimation (false upward signals)
        
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")
            
        error = pred - true
        
        # Different penalties for positive and negative errors
        pos_error = torch.where(error > 0, error, torch.zeros_like(error))
        neg_error = torch.where(error < 0, error, torch.zeros_like(error))
        
        # Asymmetric penalty
        loss = self.alpha * torch.mean(torch.abs(neg_error)) + self.beta * torch.mean(torch.abs(pos_error))
        
        return loss


class QuantileLoss(nn.Module):
    """
    Quantile loss for uncertainty quantification.
    Provides prediction intervals instead of point estimates.
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")
            
        total_loss = 0
        
        for i, q in enumerate(self.quantiles):
            error = true - pred
            loss_q = torch.mean(torch.max(q * error, (q - 1) * error))
            total_loss += loss_q
            
        return total_loss / len(self.quantiles)


class SharpeRatioLoss(nn.Module):
    """
    Loss function that directly optimizes for Sharpe ratio.
    Maximizes risk-adjusted returns.
    """
    def __init__(self, risk_free_rate=0.0, eps=1e-8):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")
            
        # Calculate returns from predictions
        if pred.dim() == 3:  # [batch, seq_len, features]
            pred_returns = pred[:, 1:] - pred[:, :-1]
        else:  # [batch, seq_len]
            pred_returns = pred[:, 1:] - pred[:, :-1]
            
        # Calculate mean and std of returns
        mean_return = torch.mean(pred_returns)
        std_return = torch.std(pred_returns)
        
        # Sharpe ratio (we want to maximize, so we minimize negative)
        sharpe_ratio = (mean_return - self.risk_free_rate) / (std_return + self.eps)
        
        return -sharpe_ratio  # Negative because we want to maximize


class TradingLoss(nn.Module):
    """
    Trading-specific loss that considers transaction costs and directional accuracy.
    """
    def __init__(self, transaction_cost=0.001, direction_weight=0.5):
        super().__init__()
        self.transaction_cost = transaction_cost
        self.direction_weight = direction_weight
        self.base_loss = nn.L1Loss()
        
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")
            
        # Base prediction loss
        base = self.base_loss(pred, true)
        
        # Directional accuracy loss
        if pred.dim() == 3:
            pred_flat = pred.squeeze(-1)
            true_flat = true.squeeze(-1)
        else:
            pred_flat = pred
            true_flat = true
            
        if pred_flat.shape[1] < 2:
            return base
            
        pred_diff = pred_flat[:, 1:] - pred_flat[:, :-1]
        true_diff = true_flat[:, 1:] - true_flat[:, :-1]
        
        # Direction mismatch penalty
        direction_mismatch = (torch.sign(pred_diff) != torch.sign(true_diff)).float()
        direction_loss = direction_mismatch.mean()
        
        # Transaction cost penalty (encourage fewer trades)
        trade_signals = torch.abs(torch.sign(pred_diff))
        transaction_loss = trade_signals.mean() * self.transaction_cost
        
        total_loss = ((1 - self.direction_weight) * base + 
                     self.direction_weight * direction_loss + 
                     transaction_loss)
        
        return total_loss


class RobustLoss(nn.Module):
    """
    Robust loss function that is less sensitive to outliers.
    Uses Huber loss with adaptive threshold.
    """
    def __init__(self, delta=1.0, adaptive=True):
        super().__init__()
        self.delta = delta
        self.adaptive = adaptive
        
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")
            
        error = pred - true
        
        if self.adaptive:
            # Adaptive threshold based on error distribution
            delta = torch.quantile(torch.abs(error), 0.75)
        else:
            delta = self.delta
            
        # Huber loss
        is_small_error = torch.abs(error) <= delta
        small_error_loss = 0.5 * error ** 2
        large_error_loss = delta * (torch.abs(error) - 0.5 * delta)
        
        loss = torch.where(is_small_error, small_error_loss, large_error_loss)
        
        return torch.mean(loss)
