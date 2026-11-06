"""Problem 1"""
from pathlib import Path
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import save_results, save_table

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hw_data"
SEED = 263

# Access data and store in a pandas data frame
df = pd.read_parquet(DATA_DIR / "taxi_trips.parquet")

####################################################################
#
#                          Problem 1(c)
#
####################################################################
TARGET = "trip_duration"

# We model log(trip_duration), since we don't want to give probabilities of trip_duration < 0
y = np.log(df[TARGET].to_numpy(dtype=np.float64))

# Isolate the features and define a trip distance measure
X = df.drop(columns=[TARGET])
X["dlon"] = df["dropoff_longitude"] - df["pickup_longitude"]
X["dlat"] = df["dropoff_latitude"] - df["pickup_latitude"]
X["dist"] = np.sqrt(X["dlon"]**2 + X["dlat"]**2)

# Split the data into 70%/15%/15% train/validation/test
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
X_va, X_te, y_va, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)

# Standardize the new data based on the training data set
x_mean, x_std = X_tr.mean(), X_tr.std()
X_tr, X_va, X_te = ((d - x_mean) / x_std for d in (X_tr, X_va, X_te))
X_tr, X_va, X_te = (d.to_numpy(np.float32) for d in (X_tr, X_va, X_te))

# Clip largest feature outliers so they can't generate extreme mean/variance predictions 
X_tr, X_va, X_te = (np.clip(d, -6, 6) for d in (X_tr, X_va, X_te))

y_mean, y_std = y_tr.mean(), y_tr.std()

#Helper function to judge mean/variance from predictions
def evaluation_function(mu: np.ndarray, var: np.ndarray, y_true: np.ndarray) -> dict:
    # Standardize residuals
    z = (y_true - mu) / np.sqrt(var)
    
    out = {
        "NLL": np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * z**2), #<--- negative log-likelihood from (b)
        "RMSE": np.sqrt(np.mean((y_true - mu) ** 2)), #<--- root mean square error (on prediction mean)
        "MAE": np.mean(np.abs(y_true - mu)), #<--- mean absolute error (on prediction mean)
        "cover_68": np.mean(np.abs(z) <= 1.0), #<--- 68% of data should be within 1 sigma of mean
        "cover_95": np.mean(np.abs(z) <= 1.96), #<--- 95% of data should be within ~2 sigma of mean
    }

    # Translates the prediction accuracy back into units of minutes
    out["MAE_minutes"] = np.mean(np.abs(np.exp(y_true) - np.exp(mu)))
    
    return out

# Dictionaries of test results and predictions
results = {}
predictions = {}

# Now, let's define some baseline models to compare the MLP to
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

# Estimate variance from the mean squared validation residual
def variance_from_validation(mu_val: np.ndarray) -> float:
    return np.mean((y_va - mu_val) ** 2)

# Constant Baseline: predict the same mean and variance for every trip (essentially, data-insensitive)
mu_te = np.full_like(y_te, y_mean)
var_te = np.full_like(y_te, y_tr.var())
predictions["constant_baseline"] = (mu_te, var_te)
results["constant_baseline"] = evaluation_function(mu_te, var_te, y_te)

# Linear Regression baseline: check whether a linear model is enough to predict this data
lin = LinearRegression().fit(X_tr, y_tr)
mu_te = lin.predict(X_te)
var_te = np.full_like(y_te, variance_from_validation(lin.predict(X_va)))
predictions["linreg_baseline"] = (mu_te, var_te)
results["linreg_baseline"] = evaluation_function(mu_te, var_te, y_te)

# Gradient Boosting baseline: Gradient Boosted Decision Trees are standard for tabular data sets
hgb = HistGradientBoostingRegressor(max_iter=500, early_stopping=True, random_state=SEED).fit(X_tr, y_tr)
mu_te = hgb.predict(X_te)
var_te = np.full_like(y_te, variance_from_validation(hgb.predict(X_va)))
predictions["gradboost_baseline"] = (mu_te, var_te)
results["gradboost_baseline"] = evaluation_function(mu_te, var_te, y_te)

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(SEED)

# Convert standardized data into tensors for PyTorch
X_tr_t, X_va_t, X_te_t = (torch.from_numpy(a) for a in (X_tr, X_va, X_te))

# Standardize y so network's outputs at initialization are on the correct scale
y_tr_t = torch.from_numpy(((y_tr - y_mean) / y_std).astype(np.float32))
y_va_t = torch.from_numpy(((y_va - y_mean) / y_std).astype(np.float32))

# Define the neural architecture g from part (a)
# Inherit from nn.Module to get PyTorch tools
class MLP(nn.Module):
    # Default constructer: d_in=input dimension, hidden=network architecture
    def __init__(self, d_in: int, hidden: Tuple[int, ...] = (128, 128, 64)):
        super().__init__()
        layers, d = [], d_in

        # Loop through the specified architecture and build the hidden layers
        for h in hidden:
            # nn.Linear is the map z -> Wz + b
            # nn.ReLU applies max(0,.) elementwise
            layers += [nn.Linear(d, h), nn.ReLU()] #<--- W is R^{hxd}, b is R^h
            d = h #<--- update the running dimension of the layer
        
        # Chain modules together so that each is applied in order to the previous module's output
        self.trunk = nn.Sequential(*layers) #<--- computes shared representation h(x) in R^{last layer dim}
        
        # Map h(x) from last hidden layer into (g_1(x), g_2(x))
        self.head = nn.Linear(d, 2)

    # Defines the computation that occurs on each call
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map (batch, d_in) --> (batch, d_last) --> (batch, 2)
        out = self.head(self.trunk(x))

        # Return mu, var from link functions in (a): f_mu = g_1, f_sigma = exp(g_2), so var = f_sigma^2 = exp(2 g_2)
        return out[:, 0], torch.exp(2 * out[:, 1])

# Gaussian negative log-likelihood from (b)
def gaussian_nll(mu: torch.Tensor, var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.gaussian_nll_loss(mu, y, var)

# Mean Squared-Error
def mse(mu: torch.Tensor, var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(mu, y)

# Define a training function that fits a model by minibatch SGD with early stopping on validation loss
# y_tr_t and y_va_t must be standardized targets
# return the model with weights determined from the best validation epoch
def train(model: nn.Module, loss_fn: Callable, X_tr_t: torch.Tensor, y_tr_t: torch.Tensor, X_va_t: torch.Tensor, y_va_t: torch.Tensor, epochs: int = 100, lr: float = 1e-1, batch_size: int = 1024, patience: int = 10, lr_patience: int = 3, lr_factor: float = 0.3) -> nn.Module:
    # Create the optimizer with all learnable weights in the network
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    # After testing this code, we added a learning-rate scheduler to improve convergence by reducing learning rate over time
    # lr -> lr_factor * lr after lr_patience epochs without improvement (min_lr caps the rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=1e-4)
    
    # Pair features and targets, then iterate over the pairing in batch_size chunks
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    
    best_val, best_state, bad_epochs = np.inf, None, 0
    
    # Each iteration is an epoch: train, then evaluate loss on validation set     
    for epoch in range(epochs):
        model.train() #TRAIN mode
        train_loss = 0.0 #<--- reset accumulated training loss
        # Iterate over all minibatches
        for xb, yb in loader:
            mu, var = model(xb) #<--- call MLP.forward() to get predicted mean/variance
            loss = loss_fn(mu, var, yb) #<--- compute loss
            opt.zero_grad() #<--- set current stored gradients to 0
            loss.backward() #<--- compute d/dtheta(loss) for each weight
            
            # Treating all gradients as a single vector, ensure it has norm <= 5
            # This caps the gradient step size, preventing a huge jump from destabilizing training
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            opt.step() #<--- apply update to each parameter
            train_loss += loss.item() * len(xb) #<--- update training loss with full sum over batch
        train_loss /= len(X_tr_t) #<--- compute average training loss for the epoch

        model.eval() #EVALUATION mode
        with torch.no_grad():
            mu, var = model(X_va_t) #<--- evaluate the model on validation data
            val_loss = loss_fn(mu, var, y_va_t).item() #<--- evaluate the validation loss

        sched.step(val_loss) #<--- scheduler checks validation loss and reduces lr if it has plateaued
        current_lr = opt.param_groups[0]["lr"] #<--- grab this epoch's learning rate
        print(f"epoch {epoch:3d}   train {train_loss:.4f}   val {val_loss:.4f}   lr {current_lr:.1e}")

        # Define the early stopping criteria:
        if val_loss < best_val - 1e-4:
            # If validation loss improved, record the new best value and save a copy of the weights
            best_val, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            # If validation loss did not improve, count a bad epoch
            bad_epochs += 1
            if bad_epochs >= patience:
                # After 'patience' # of bad epochs, exit: the model has stopped improving
                print(f"Early stopping at epoch {epoch}. Best validation loss = {best_val:.4f}")
                break

    # Overwrite current weights with those from the "best state"
    # This ensures we return the best generalizing model, rather than the final state
    model.load_state_dict(best_state)
    return model

# Define a prediction function that runs the trained model on X_t and returns (mu, var) in log-minutes.
# Output: predicted (mean[i], variance[i]) for each test trip i
def predict(model: nn.Module, X_t: torch.Tensor, y_mean: float, y_std: float) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        model.eval()
        mu, var = model(X_t)
        mu = mu.numpy().astype(np.float64)
        var = var.numpy().astype(np.float64)
        return mu * y_std + y_mean, var * y_std**2

# Train the network and assess its performance on the test set
p = X_tr.shape[1] #<--- number of input features
torch.manual_seed(SEED)

# MLP baseline: using the same MLP architecture, check performance if we train only with mean (i.e. MSE loss)
mlp_mse = train(MLP(p), mse, X_tr_t, y_tr_t, X_va_t, y_va_t)

# Run the trained model on validation and test sets
mu_va, _ = predict(mlp_mse, X_va_t, y_mean, y_std)
mu_te, _ = predict(mlp_mse, X_te_t, y_mean, y_std)

# We pass a single constant from validation residuals as the variance, following the other baselines.
var_te = np.full_like(y_te, variance_from_validation(mu_va))
predictions["mlp_mse_baseline"] = (mu_te, var_te)
results["mlp_mse_baseline"] = evaluation_function(mu_te, var_te, y_te)

# Train MLP with Gaussian NLL (from (b))
torch.manual_seed(SEED)
mlp_nll = train(MLP(p), gaussian_nll, X_tr_t, y_tr_t, X_va_t, y_va_t, epochs=150)
mu_te, var_te = predict(mlp_nll, X_te_t, y_mean, y_std)
predictions["mlp_gaussianNLL"] = (mu_te, var_te)
results["mlp_gaussianNLL"] = evaluation_function(mu_te, var_te, y_te)

# Compare all models on the test set and output results
print("Results from applying all models on the test set: ")
print(pd.DataFrame(results).T.round(4))

# Save the test-set results for the report
MODEL_NAMES = {"constant_baseline": "Constant", "linreg_baseline": "Linear regression", "gradboost_baseline": "Gradient boosting", "mlp_mse_baseline": "MLP (MSE loss)", "mlp_gaussianNLL": "MLP (Gaussian NLL)"}
save_results("problem1_test_metrics", {name: {k: float(v) for k, v in r.items()} for name, r in results.items()})
save_table("p1_metrics_table", ["Model", "NLL", "RMSE", "MAE", "MAE (min)", "Cover 68\\%", "Cover 95\\%"], [[MODEL_NAMES[name], f"{r['NLL']:.3f}", f"{r['RMSE']:.3f}", f"{r['MAE']:.3f}", f"{r['MAE_minutes']:.2f}", f"{r['cover_68']:.3f}", f"{r['cover_95']:.3f}"] for name, r in results.items()], "lrrrrrr")
####################################################################

####################################################################
#
#                          Problem 1(d)
#
####################################################################
from scipy.stats import norm

# Helper function: Prob(duration < threshold | x), presuming log Y | x ~ Normal(mu(x), Var(x))
def prob_shorter_than(mu: np.ndarray, var: np.ndarray, threshold_minutes: float = 45.0) -> np.ndarray:
    return norm.cdf((np.log(threshold_minutes) - mu) / np.sqrt(var)) #<--- P(log Y < log 45) from (d)
####################################################################

####################################################################
#
#                          Problem 1(e)
#
####################################################################
# Label trips that actually took < 45 min as 'True'
actual = np.exp(y_te) < 45

# Estimate P(trip < 45 min) on the test set for each model, then score the probabilities
# Brier score = mean squared error of the probabilities, log-loss = negative log-likelihood of the observed labels
# Output "< 45 min" whenever P >= 50% to get the classification accuracy
accuracy, brier, logloss = {}, {}, {}
for name, (mu, var) in predictions.items():
    p45_te = prob_shorter_than(mu, var) #<--- predicted probability for each test trip
    p_clip = np.clip(p45_te, 1e-12, 1 - 1e-12) #<--- avoid log(0) for overconfident predictions
    predicted = p45_te >= 0.5
    accuracy[name] = np.mean(predicted == actual)
    brier[name] = np.mean((p45_te - actual) ** 2)
    logloss[name] = -np.mean(actual * np.log(p_clip) + (1 - actual) * np.log(1 - p_clip))

baseline_accuracy = np.mean(actual)

print(f"Accuracy of always predicting '< 45 min' on the test set: {baseline_accuracy:.4f}")
print("Scores from all models: ")
print(pd.DataFrame({"Brier": brier, "Log-loss": logloss, "Accuracy": accuracy}).round(4))

# Save the probability scores for the report
# Always predicting '< 45 min' means p = 1 for every trip, so its Brier score is the error rate and its log-loss is infinite
save_results("problem1_prob_scores", {"always_below_45": {"accuracy": float(baseline_accuracy), "brier": float(1 - baseline_accuracy)}, **{name: {"brier": float(brier[name]), "log_loss": float(logloss[name]), "accuracy": float(accuracy[name])} for name in predictions}})
save_table("p1_prob_table", ["Model", "Brier score", "Log-loss", "Accuracy"], [["Always predict $<$45 min", f"{1 - baseline_accuracy:.4f}", "--", f"{baseline_accuracy:.4f}"]] + [[MODEL_NAMES[name], f"{brier[name]:.4f}", f"{logloss[name]:.4f}", f"{accuracy[name]:.4f}"] for name in predictions], "lrrr")
####################################################################
