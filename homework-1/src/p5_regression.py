"""Problem 5 (regression)"""
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_ale, save_results, save_table

# Degrees of freedom in t-distribution
NU = 5.0 

X, y = load_ale()

# Random split into 80% train, 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

# First, we perform a basic Linear Regression
# The logic mirrors the setup in p5_classification.py
baseline = make_pipeline(StandardScaler(), LinearRegression())
baseline.fit(X_train, y_train)
baseline_mae = mean_absolute_error(y_test, baseline.predict(X_test))
print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
print(f"sklearn LinearRegression Test MAE: {baseline_mae:.4f}")

# Borrow the scaled train/test data from the pipeline and convert NumPy arrays into PyTorch tensors
scaler = baseline[0]
X_train_t = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
X_test_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
n_train, n_features = X_train_t.shape #[0] = training rows, [1] = number of features

# (a) Define the loss function for the Linear Regression model
def mse_loss(pred, y):
    # Return the mean (over all observations) so that we have a singular quantity to minimize
    return ((y - pred) ** 2).mean()

# (b) Define the negative log-likelihood loss function for the t-distribution model
def student_t_nll(pred, y):
    # Return the mean (over all observations) so that we have a singular quantity to minimize
    return -torch.distributions.StudentT(df=NU, loc=pred, scale=1.0).log_prob(y).mean()


# We can use a similar training function as p5_classification.py
# The difference is that the model (Linear Regression) is the same in (a) and (b),
# what changes is the loss we are minimizing. We pass *loss_fn* as a parameter instead of *model*
# With only 22 test points, we default batch_size = 16
def train_sgd(loss_fn, lr=0.1, n_epochs=100, batch_size=16):
    torch.manual_seed(7)
    w = torch.zeros(n_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=lr)
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]

            # Difference: minimize desired loss function with Linear Regression model
            loss = loss_fn(torch.matmul(X_train_t[idx], w) + b, y_train_t[idx])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                full_loss = loss_fn(torch.matmul(X_train_t, w) + b, y_train_t).item()
            print(f"  epoch {epoch:3d}  train loss {full_loss:.4f}")
    return w.detach(), b.detach()

print("(a) Linear regression (SGD, Loss: MSE):")
w_mse, b_mse = train_sgd(mse_loss, n_epochs=200, lr=0.05)
print(f"(b) Linear regression (SGD, Loss: Student-t NLL, nu={NU:g}):")
w_t, b_t = train_sgd(student_t_nll, n_epochs=200, lr=0.05)

# Function to determine mean absolute error of model when applied to test data
def test_mae(w, b):
    with torch.no_grad():
        return (torch.matmul(X_test_t, w) + b - y_test_t).abs().mean().item()

# Compute mean absolute error for comparison with baseline 
maes = {
    "sklearn LinearRegression": baseline_mae,
    "SGD, MSE loss": test_mae(w_mse, b_mse),
    f"SGD, Student-$t$ NLL ($\\nu={NU:g}$)": test_mae(w_t, b_t),
}

# Save results
print(f"{'Method':<28}{'Test MAE':>10}")
rows = []
for name, mae in maes.items():
    print(f"{name:<28}{mae:>10.4f}")
    rows.append([name, f"{mae:.4f}"])

save_results("problem5_regression", {k: {"test_mae": v} for k, v in maes.items()})
save_table("p5_reg_table", ["Method", "Test MAE"], rows, "lr")
