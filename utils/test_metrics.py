import torch
from metrics import MDAMetric, SharpeRatioMetric, GMADLLoss, MADLLoss, DLFLoss

# Closer to 1 is better in MDA
mda_fn = MDAMetric()

# Create dummy prediction and true returns
pred = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
true = torch.tensor([[0.1, 0.3, 0.2, 0.5]])
mda_val = mda_fn(pred, true).item()
print(f"MDA: {mda_val}")
assert 0.0 <= mda_val <= 1.0

# same direction
pred = torch.tensor([[1.0, 2.0, 3.0]])
true = torch.tensor([[0.5, 1.5, 2.5]])
mda_val = mda_fn(pred, true).item()
assert mda_val == 1.0
print(f"MDA: {mda_val}")

# opposite direction
pred = torch.tensor([[3.0, 2.0, 1.0]])
true = torch.tensor([[0.5, 1.5, 2.5]])
mda_val = mda_fn(pred, true).item()
assert mda_val == 0.0
print(f"MDA: {mda_val}")


# Higher values are better in Sharpe
sharpe_fn = SharpeRatioMetric()

# Positive returns
pred_pos = torch.tensor([[0.02, 0.03, 0.01]])
sharpe_pos = sharpe_fn(pred_pos, pred_pos).item()
print(f"Sharpe (positive returns): {sharpe_pos}")

# Negative returns
pred_neg = torch.tensor([[-0.02, -0.03, -0.01]])
sharpe_neg = sharpe_fn(pred_neg, pred_neg).item()
print(f"Sharpe (negative returns): {sharpe_neg}")

# Increasing returns should generally raise Sharpe
pred_mix = torch.tensor([[0.01, 0.02, 0.03]])
sharpe_mix = sharpe_fn(pred_mix, pred_mix).item()
print(f"Sharpe (increasing returns): {sharpe_mix}")


# Negative values are better in GMADL
gmadl_fn = GMADLLoss(a=1.0, b=1.0)

# Should be low
pred = torch.tensor([[0.02, 0.03]])
true = torch.tensor([[0.02, 0.03]])
val = gmadl_fn(pred, true).item()
print(f"GMADL: {val}")

# Should be higher
pred = torch.tensor([[0.02, 0.03]])
true = torch.tensor([[-0.02, -0.03]])
val = gmadl_fn(pred, true).item()
print(f"GMADL: {val}")

# Should be zero
pred = torch.tensor([[0.0, 0.0]])
true = torch.tensor([[0.0, 0.0]])
val = gmadl_fn(pred, true).item()
print(f"GMADL: {val}")


# Negative values are better in MADL
madl_fn = MADLLoss()

pred_returns = torch.tensor([[0.01, 0.03, 0.04]])
true_returns = torch.tensor([[0.01, 0.03, 0.04]])
loss = madl_fn(pred_returns, true_returns)
print(f"MADL: {loss.item()}")

pred_returns = torch.tensor([[0.01, 0.03, 0.04]])
true_returns = torch.tensor([[-0.01, -0.03, -0.04]])
loss = madl_fn(pred_returns, true_returns)
print(f"MADL: {loss.item()}")

pred_returns = torch.tensor([[0.01, 0.02, -0.01]])
true_returns = torch.tensor([[0.01, -0.02, -0.01]])
loss = madl_fn(pred_returns, true_returns)
print(f"MADL: {loss.item()}")

# DLF Testing

from metrics import DLFLoss

dlf_fn = DLFLoss(lambda_weight=0.5)

# Directionally aligned
pred = torch.tensor([[1.0, 2.0, 3.0]])
true = torch.tensor([[1.5, 2.5, 3.5]])
print("DLF aligned:", dlf_fn(pred, true).item())

# Directionally opposite
pred = torch.tensor([[3.0, 2.0, 1.0]])
true = torch.tensor([[1.5, 2.5, 3.5]])
print("DLF opposite:", dlf_fn(pred, true).item())
