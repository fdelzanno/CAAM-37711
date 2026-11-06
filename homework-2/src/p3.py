"""Problem 3"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from common import FIGURES_DIR, RESULTS_DIR, save_results, save_table

@dataclass
class Split:
    X: np.ndarray
    y: Optional[np.ndarray] = None  # None for the unlabeled test split

@dataclass
class HW2Dataset:
    train: Split
    val: Split
    test: Split

    @classmethod
    def from_csv(cls, train_path: str, val_path: str, test_path: str):
        train_data = np.loadtxt(train_path, delimiter=",", skiprows=1)
        val_data = np.loadtxt(val_path, delimiter=",", skiprows=1)
        test_data = np.loadtxt(test_path, delimiter=",", skiprows=1)

        train_split = Split(train_data[:, :2], train_data[:, 2])
        val_split = Split(val_data[:, :2], val_data[:, 2])
        # test.csv contains only x1, x2 -- the labels are held out for grading
        test_split = Split(test_data[:, :2], None)

        return cls(train_split, val_split, test_split)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hw_data"
SEED = 73

####################################################################
#
#                          Problem 3(a)
#
####################################################################

ds = HW2Dataset.from_csv(f"{DATA_DIR}/train.csv", f"{DATA_DIR}/val.csv", f"{DATA_DIR}/test.csv")
print(ds.train.X.shape, ds.train.y.shape)
print(ds.val.X.shape, ds.val.y.shape)
print(ds.test.X.shape, ds.test.y)

x1, x2 = ds.train.X[:, 0], ds.train.X[:, 1]
y = ds.train.y

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
sc = ax1.scatter(x1, x2, c=y, cmap="viridis", s=15)
fig.colorbar(sc, ax=ax1, label="y")
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_aspect("equal")
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_title("Training Data")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(x1, x2, y, c=y, cmap="viridis", s=10)
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_zlabel("y")
ax2.set_title("Training Data")

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "p3_training_data.png", dpi=200)
plt.show()
####################################################################

####################################################################
#
#                          Problem 3(b)
#
####################################################################
from typing import List, Tuple
import torch
import torch.nn as nn

"""
For Random Kitchen Sinks (RKS)
"""
class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3, seed: int=SEED):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            D: total number of random features (= 2h in the assignment text; if you use both sin and cos features, D should be an even number)
            sigma: scale parameter for the RBF kernel, used for initializing the weight parameters
            ridge_lambda: ridge regularization strength
        """
        self.D = D  # total feature count
        self.h = D // 2 # number of random projections (for RKS)
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda
        self.random = np.random.default_rng(seed)   # random number generator (for reproducibility)
        self.W = None
        self.b = None
        self.coef_ = None # fitted ridge coefficients (D+1,) [last entry is intercept term]

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Convert the input data X into the random kitchen sinks features. You will likely need to use the parameters initialized in the __init__ method.
        The resulting features should be of shape (n, D or D[+1]), where the plus one is for the intercept term.
        """
        rks_features = np.matmul(X, self.W) + self.b
        ones = np.ones((X.shape[0], 1))

        # Apply sin/cos elementwise and then concatenate column-wise [(n,h) + (n,h) + (n,1) = (n,D+1)] 
        return np.hstack([np.sin(rks_features), np.cos(rks_features), ones])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ridge regression model to the (1) features computed from the input data X and (2) target values y.
        """
        d = X.shape[1]  # Get input dimension

        # Initially: randomize weights and bias
        self.W = self.random.normal(0.0, 1.0 / self.sigma, size=(d, self.h))
        self.b = self.random.uniform(0.0, 2 * np.pi, size=self.h)
        
        phi = self._features(X)  # design matrix (n, D+1)

        # Build the penalty matrix, then remove the penalty on the intercept
        reg = self.ridge_lambda * np.eye(phi.shape[1])
        reg[-1, -1] = 0.0   #shrinking the offset of y toward zero doesn't matter

        # Solve (phi^T * phi + reg) * coef_ = phi^T * y for coef_
        self.coef_ = np.linalg.solve(np.matmul(phi.T, phi) + reg, np.matmul(phi.T, y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data X.
        """
        phi = self._features(X)  # turn a new point into features [shape: (n, D+1)]
        return np.matmul(phi, self.coef_)  # take a weighted sum with the ridge weights [shape: (n,)]

def sweep_rks_model(sweep_D: List[int], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, sigma: float = 1.0, ridge_lambda: float = 1e-3) -> Tuple:
    """
    An example sweeping function that sweeps over the number of random features D, it then returns the best model in terms of validation MSE.
    Args:
        sweep_D: list of numbers of random features to sweep over
        X_train: training input data
        y_train: training target values
        X_val: validation input data
        y_val: validation target values
        sigma: scale parameter for the RBF kernel, used for initializing the weight parameters
        ridge_lambda: ridge regularization strength
    Returns:
        a tuple (best_model, best_val_mse, best_D, train_mses, val_mses)
    """
    best_model = None
    best_val_mse = float('inf')
    best_D = None
    train_mses, val_mses = [], []

    for D in sweep_D:
        # Build a new model for each D (with sigma, ridge_lambda fixed)
        model = RandomKitchenSinks(D=D, sigma=sigma, ridge_lambda=ridge_lambda).fit(X_train, y_train)
        
        train_mse = mse(y_train, model.predict(X_train)) # Use MSE to measure fit quality
        train_mses.append(train_mse)
        
        val_mse = mse(y_val, model.predict(X_val)) # Use MSE to estimate generalization
        val_mses.append(val_mse)
        print(f"D = {D:5d} | train MSE = {train_mse:.4f} | val MSE = {val_mse:.4f}")

        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_D = D

    return best_model, best_val_mse, best_D, train_mses, val_mses

# Helper function that returns the average squared residual (mean squared error)
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred)**2)

sweep_D = [10, 20, 50, 100, 200, 500, 1000, 2000]
best_rks, best_val_mse, best_D, train_mses, val_mses = sweep_rks_model(sweep_D, ds.train.X, ds.train.y, ds.val.X, ds.val.y)
print(f"\nBest D = {best_D} with Validation MSE = {best_val_mse:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(sweep_D, train_mses, "o-")
axes[0].set_title("Random Kitchen Sink: Training MSE")

axes[1].plot(sweep_D, val_mses, "o-", color="tab:orange")
axes[1].axvline(best_D, ls="--", color="gray", label=f"Best D = {best_D}")
axes[1].set_title("Random Kitchen Sink: Validation MSE")
axes[1].legend()

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("Number of Random Features, D")
    ax.set_ylabel("MSE")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "p3_rks_sweep.png", dpi=200)
plt.show()

# Save the sweep results for the report
save_results("problem3_rks_sweep", {"D": sweep_D, "train_mse": [float(m) for m in train_mses], "val_mse": [float(m) for m in val_mses], "best_D": best_D})
save_table("p3_rks_sweep_table", ["$D$", "Train MSE", "Validation MSE"], [[f"{D:,}", f"{tr:.4f}", f"{va:.4f}"] for D, tr, va in zip(sweep_D, train_mses, val_mses)], "rrr")
####################################################################

####################################################################
#
#                          Problem 3(c)
#
####################################################################
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    """1-hidden-layer MLP for regression."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3, batch_size: int = 128, n_epochs: int = 500, activation: str = "relu", device: str = "cpu", seed: int = SEED):
        super().__init__()  # inherit nn.Module from PyTorch
        self.device = device
        self.lr = lr    # SGD step size
        self.batch_size = batch_size    # number of samples per SGD step
        self.n_epochs = n_epochs    # number of passes over the training set
        
        self.seed = seed
        torch.manual_seed(seed)

        # Allow different activation functions, since we pass 'activation' as a parameter
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}

        # 1 hidden-layer architecture: x (d_in) -> linear -> nonlinearity -> linear -> y_hat (d_out)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),  # learned weights + biases [shape: (d_in -> d_hidden)]
            activations[activation],    # elementwise nonlinearity
            nn.Linear(d_hidden, d_out), # output layer is weighted sum of hidden features + intercept
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Convert data to torch tensors [reshape y from (n,) to (n, 1) to match NN output]
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        # Pair each X row with its y value, then split into shuffled mini-batches
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr) # SGD: param = param - lr * gradient
        loss_fn = nn.MSELoss()  # evaluate MSE loss (same as RKS)

        self.net.train()    # make sure to set the model to training mode
        for epoch in range(self.n_epochs):
            # Loop over mini-batches
            for X_batch, y_batch in loader:
                optimizer.zero_grad()                          # clear gradients from the previous step
                loss = loss_fn(self.forward(X_batch), y_batch) # compute MSE loss on batch
                loss.backward()                                # compute d(loss)/d(param) for every parameter and backpropagate 
                optimizer.step()                               # update parameters with calculated gradients
        return self

    # Helper function: passes the input through the NN [Starts as (n, d_in), ends as (n, 1)]
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval() # go into evaluation mode
        
        # Don't track gradients during the predictive step
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)

            # self.forward(X) runs input X through the trained network
            return self.forward(X).cpu().numpy().ravel()

# This sweeping function is identical to the RKS sweeping function, but configured for the MLP and sweeps over hidden layer dimensions.
def mlp_sweep(sweep_values: List[int], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, lr: float = 1e-3, batch_size: int = 128, n_epochs: int = 500, activation: str = "relu") -> Tuple:
    best_model = None
    best_val_mse = float('inf')
    best_hidden_dim = None
    train_mses, val_mses = [], []

    for d_hidden in sweep_values:
        # Build a new model for each hidden layer dimension (fixing all other parameters)
        model = MLP(d_in=X_train.shape[1], d_hidden=d_hidden, lr=lr, batch_size=batch_size, n_epochs=n_epochs, activation=activation).fit(X_train, y_train)

        train_mse = mse(y_train, model.predict(X_train))    # Use MSE to measure fit quality
        train_mses.append(train_mse)

        val_mse = mse(y_val, model.predict(X_val))  # Use MSE to estimate generalization
        val_mses.append(val_mse)
        print(f"d_hidden = {d_hidden:5d} | train MSE = {train_mse:.4f} | val MSE = {val_mse:.4f}")

        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_hidden_dim = d_hidden

    return best_model, best_val_mse, best_hidden_dim, train_mses, val_mses

sweep_hidden_dim = [2, 4, 8, 16, 32, 64, 128, 256, 512]
best_mlp, best_mlp_val_mse, best_hidden_dim, mlp_train_mses, mlp_val_mses = mlp_sweep(sweep_hidden_dim, ds.train.X, ds.train.y, ds.val.X, ds.val.y, lr=1e-2, n_epochs=2000)
print(f"\nBest dimension of hidden layer = {best_hidden_dim} with Validation MSE = {best_mlp_val_mse:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(sweep_hidden_dim, mlp_train_mses, "o-")
axes[0].set_title("MLP: Training MSE")

axes[1].plot(sweep_hidden_dim, mlp_val_mses, "o-", color="tab:orange")
axes[1].axvline(best_hidden_dim, ls="--", color="gray", label=f"Best hidden layer dimension = {best_hidden_dim}")
axes[1].set_title("MLP: Validation MSE")
axes[1].legend()

for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Dimension of Hidden Layer")
    ax.set_ylabel("MSE")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "p3_mlp_sweep.png", dpi=200)
plt.show()

# Save the sweep results for the report
save_results("problem3_mlp_sweep", {"d_hidden": sweep_hidden_dim, "train_mse": [float(m) for m in mlp_train_mses], "val_mse": [float(m) for m in mlp_val_mses], "best_hidden_dim": best_hidden_dim})
save_table("p3_mlp_sweep_table", ["$d_{\\mathrm{hidden}}$", "Train MSE", "Validation MSE"], [[f"{d:,}", f"{tr:.4f}", f"{va:.4f}"] for d, tr, va in zip(sweep_hidden_dim, mlp_train_mses, mlp_val_mses)], "rrr")
####################################################################

####################################################################
#
#                          Problem 3(d)
#
####################################################################
from typing import Callable

def plot_surface_2d(model_fn: Callable[[np.ndarray], np.ndarray] | nn.Module, xlim: Tuple[float, float], ylim: Tuple[float, float], title: str = "Fitted surface", filename: str = "surface.png", grid: int = 200):
    """
        An example plotting function that handles both callable functions (e.g. model.predict) for a sk-learn-like model and a PyTorch NN.
    
        Example usage:
        >>> # some model with a `.predict` method
        >>> rks_model = fit_rks_model(X_train, y_train)
        >>> plot_surface_2d(rks_model.predict, (-3, 3), (-3, 3), title="RKS surface")
        >>> # some PyTorch model
        >>> mlp_model = fit_mlp_model(X_train, y_train)
        >>> # direct feed it to the plotting function, no need to call `.predict`
        >>> plot_surface_2d(mlp_model, (-3, 3), (-3, 3), title="MLP surface")
    """
    
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X1, X2 = np.meshgrid(xs, ys)
    Xgrid = np.stack([X1.ravel(), X2.ravel()], axis=1)
    if isinstance(model_fn, nn.Module):
        with torch.no_grad():
            X_grid = torch.from_numpy(Xgrid).float().to(next(model_fn.parameters()).device)
            Z = model_fn(X_grid).cpu().numpy().reshape(grid, grid)
    else:
        Z = model_fn(Xgrid).reshape(grid, grid)

    plt.figure()
    plt.contourf(X1, X2, Z, levels=30)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(label="pred")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

# Refit both models with the best hyperparameters from (b) and (c)
rks_final = RandomKitchenSinks(D=best_D, sigma=1.0, ridge_lambda=1e-3).fit(ds.train.X, ds.train.y)
mlp_final = MLP(d_in=ds.train.X.shape[1], d_hidden=best_hidden_dim, lr=1e-2, n_epochs=2000).fit(ds.train.X, ds.train.y)

rks_val_mse = mse(ds.val.y, rks_final.predict(ds.val.X))
mlp_val_mse = mse(ds.val.y, mlp_final.predict(ds.val.X))
print(f"RKS (D = {best_D}):           Validation MSE = {rks_val_mse:.4f}")
print(f"MLP (d_hidden = {best_hidden_dim}):     Validation MSE = {mlp_val_mse:.4f}")

# Save the final comparison for the report
save_results("problem3_final_val_mse", {"rks": {"D": best_D, "val_mse": float(rks_val_mse)}, "mlp": {"d_hidden": best_hidden_dim, "val_mse": float(mlp_val_mse)}})
save_table("p3_comparison_table", ["Model", "Best hyperparameter", "Validation MSE"], [["RKS (ridge)", f"$D = {best_D}$", f"{rks_val_mse:.4f}"], ["MLP (1 hidden layer)", f"$d_{{\\mathrm{{hidden}}}} = {best_hidden_dim}$", f"{mlp_val_mse:.4f}"]], "llr")

# Save and output predictions on the unlabeled test set
OUTPUT_DIR = RESULTS_DIR
rks_test_pred = rks_final.predict(ds.test.X)
mlp_test_pred = mlp_final.predict(ds.test.X)

np.savetxt(OUTPUT_DIR / "rks_test_predictions.csv", rks_test_pred, delimiter=",")
np.savetxt(OUTPUT_DIR / "mlp_test_predictions.csv", mlp_test_pred, delimiter=",")

# Plot fitted surfaces
plot_surface_2d(rks_final.predict, (-3, 3), (-3, 3), title=f"RKS fitted surface (D = {best_D})", filename=f"{FIGURES_DIR}/p3_rks_surface.png")
plot_surface_2d(mlp_final, (-3, 3), (-3, 3), title=f"MLP fitted surface (d_hidden = {best_hidden_dim})", filename=f"{FIGURES_DIR}/p3_mlp_surface.png")
plt.show()
####################################################################