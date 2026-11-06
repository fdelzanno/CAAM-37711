"""Problem 2"""
from pathlib import Path
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import save_results, save_table

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hw_data"
SEED = 73

# Access data and store in a pandas data frame
df = pd.read_csv(DATA_DIR / "retail.csv")

####################################################################
#
#                          Problem 2(a)
#
####################################################################

# Convert categorical features to categorical data types (for encoding)
df["CustomerID"] = df["CustomerID"].astype(np.int64).astype("category")
df["StockCode"] = df["StockCode"].astype("category")
feature_cols = ["StockCode", "CustomerID"]

print(df.head())

cardinality = {}
for col in feature_cols:
    print("\n")
    print(col, "unique values:", df[col].nunique()) #<--- how many unique values are there?
    
    counts = df[col].value_counts() #<--- in how many rows does each unique category appear?
    print(f"Summary of rows per unique {col}: median={counts.median():.0f}, min={counts.min()}, max={counts.max()}")
    
    print(f"{col} values appearing in <= 5 rows: {(counts <= 5).sum()}") #<--- how many categories appear in <= 5 rows? (i.e. "rare")
    cardinality[col] = {"unique": int(df[col].nunique()), "median": float(counts.median()), "min": int(counts.min()), "max": int(counts.max()), "rare": int((counts <= 5).sum())}

# Is one-hot encoding feasible? Calculate the size of the encoding:
n_onehot = df["StockCode"].nunique() + df["CustomerID"].nunique()
print(f"\nOne-hot columns: {n_onehot}")
onehot_gb = len(df) * n_onehot * 8 / 1e9
print(f"Dense float64 one-hot size: {onehot_gb:.1f} GB")

# Prepare the data for next steps by performing a train/validation/test split (70%/15%/15%)
X = df[feature_cols]
y = df["Quantity"]

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
print(f"\nSize of: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

# After a random split, some rare categories may only be in validation/test, so we count how often that happens
unseen_counts = {}
for col in feature_cols:
    train_categories = set(X_train[col].unique())
    unseen_counts[col] = {}
    for name, Xs in [("Validation", X_val), ("Test", X_test)]:
        unseen = ~Xs[col].isin(train_categories)
        unseen_counts[col][name] = int(unseen.sum())
        print(f"{col} {name}: {unseen.sum()} rows with categories unseen in train")

# Save the results for use in the report
save_results("problem2_exploration", {"n_rows": len(df), "onehot_columns": int(n_onehot), "onehot_gb": float(onehot_gb), "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}, "cardinality": cardinality, "unseen_rows": unseen_counts})
save_table("p2_cardinality_table", ["Feature", "Unique values", "Median rows", "Min rows", "Max rows", "Values in $\\le 5$ rows"], [[col, f"{c['unique']:,}", f"{c['median']:.0f}", f"{c['min']}", f"{c['max']:,}", f"{c['rare']:,}"] for col, c in cardinality.items()], "lrrrrr")
save_table("p2_unseen_table", ["Feature", "Validation rows", "Test rows"], [[col, f"{u['Validation']:,}", f"{u['Test']:,}"] for col, u in unseen_counts.items()], "lrr")
####################################################################

####################################################################
#
#                          Problem 2(b)
#
####################################################################
from category_encoders import TargetEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error, mean_squared_error

# Initial testing showed that two extreme orders caused the boosting to diverge 
# We therefore cap extreme training targets to stabilize the fit
cap = y_train.quantile(0.999)
y_train_cap = y_train.clip(upper=cap)
print(f"Capping training 'Quantity' at {cap:.0f} ({(y_train > cap).sum()} rows affected)")

# Perform target encoding on the training set, then apply it to all three sets
# Target encoding replaces a categorical variable with its average "effect" on the target
# Note: TargetEncoder smooths by default, which smooths rare categories' means toward the overall mean
encoder = TargetEncoder(cols=feature_cols)
X_train_te = encoder.fit_transform(X_train, y_train_cap)
X_val_te = encoder.transform(X_val)
X_test_te = encoder.transform(X_test)

# Verify that each category is replaced by a mean 'Quantity' (the target)
print(X_train_te.head())

# Perform gradient boosting
# We choose a Poisson loss function, since 'Quantity' is a non-negative count
model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=500, early_stopping=True, random_state=SEED)
model.fit(X_train_te, y_train_cap)

def summarize_result(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {
        "deviance": mean_poisson_deviance(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    print(f"\n{name}: ")
    print(f"Poisson deviance={out['deviance']:.2f}")
    print(f"MAE={out['MAE']:.2f}")
    print(f"RMSE={out['RMSE']:.2f}")
    return out

# Define a baseline for comparison: always predict the training-set mean
metrics = {"Baseline": {}, "Target Encoding": {}, "Embeddings": {}, "Combined": {}}
metrics["Baseline"]["Train"] = summarize_result("Baseline (on training set)", y_train, np.full(len(y_train), y_train.mean()))
metrics["Baseline"]["Validation"] = summarize_result("Baseline (on validation set)", y_val, np.full(len(y_val), y_train.mean()))
metrics["Baseline"]["Test"] = summarize_result("Baseline (on test set)", y_test, np.full(len(y_test), y_train.mean()))

# Evaluate performance on train, validation, and test
metrics["Target Encoding"]["Train"] = summarize_result("Train", y_train, model.predict(X_train_te))
metrics["Target Encoding"]["Validation"] = summarize_result("Validation", y_val, model.predict(X_val_te))
metrics["Target Encoding"]["Test"] = summarize_result("Test", y_test, model.predict(X_test_te))
####################################################################

####################################################################
#
#                          Problem 2(c)
#
####################################################################
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(SEED)

mappings = {}

# Map each training category to an integer index for embedding lookup
for col in feature_cols:
    train_categories = X_train[col].unique()
    mappings[col] = {}
    for i, cat in enumerate(train_categories):
        mappings[col][cat] = i + 1  # reserve 0 for "unseen" categories

# Convert a feature data frame into a (N, 2) tensor of integer indices, where column 0 = StockCode, column 1 = CustomerID
def to_indices(X: pd.DataFrame) -> torch.Tensor:
    tensors = []
    for col in feature_cols:
        idx = X[col].astype(object).map(mappings[col]) #<--- label converts to integer (NaN if unseen)
        idx = idx.fillna(0).astype(np.int64) #<--- 'unseen' converted to integer 0
        tensors.append(torch.tensor(idx.values))
    return torch.stack(tensors, dim=1) #<--- stack (N,) and (N,) into a (N, 2) tensor

X_tr_t = to_indices(X_train) 
X_va_t = to_indices(X_val)
X_te_t = to_indices(X_test)

# As in (b), we train on the capped target (but validate against the original target)
y_tr_t = torch.tensor(y_train_cap.values, dtype=torch.float32)
y_va_t = torch.tensor(y_val.values, dtype=torch.float32)

# Define the embedding network
# Inherit from nn.Module to get PyTorch tools
class EmbeddingNet(nn.Module):
    # n_stock, n_cust = number of training categories, emb_dim = embedding dimension, hidden = MLP architecture
    # n_extra = number of additional numeric input features (0 for part (c), 2 for part (d))
    def __init__(self, n_stock: int, n_cust: int, emb_dim: int = 16, hidden: Tuple[int, ...] = (64,), init_bias: float = 0.0, n_extra: int = 0):
        super().__init__()

        # One lookup table per feature: row i is the learned vector for category i
        # Row 0 is reserved for unseen categories --> padding_idx=0 fixes it at the zero vector
        self.stock_emb = nn.Embedding(n_stock + 1, emb_dim, padding_idx=0) #<--- table in R^{(n_stock+1) x emb_dim}
        self.cust_emb = nn.Embedding(n_cust + 1, emb_dim, padding_idx=0)   #<--- table in R^{(n_cust+1) x emb_dim}

        # The two embeddings are concatenated, so the MLP input has dimension 2*emb_dim + n_extra
        layers, d = [], 2 * emb_dim + n_extra

        # Loop through the specified architecture and build the hidden layers
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()] #<--- W is R^{hxd}, b is R^h
            d = h #<--- update the running dimension of the layer

        # Chain modules together so that each is applied in order to the previous module's output
        self.trunk = nn.Sequential(*layers)

        # Map the last hidden layer to a single output: the log of the predicted mean Quantity (to ensure positivity)
        self.head = nn.Linear(d, 1)

        # Start the output bias at log(training mean), so the untrained network predicts (approximately) the mean
        nn.init.constant_(self.head.bias, init_bias)

        # Initialize embeddings near zero, so the predictions for rare categories come mainly from the network's biases.
        # Since the output bias is initialized at log(training mean), this is analogous to the smoothing from (b)
        # (Empirically motivated, after seeing that random initialization spent many epochs undoing random noise)
        nn.init.normal_(self.stock_emb.weight, std=0.01)
        nn.init.normal_(self.cust_emb.weight, std=0.01)

        # Reset the unseen-category row to exactly zero, since normal_ (above) overwrote it
        with torch.no_grad():
            self.stock_emb.weight[0].zero_()
            self.cust_emb.weight[0].zero_()

    # Defines the computation that occurs on each call
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Columns 0-1 hold the integer category indices, and we let any remaining columns be extra numeric features
        idx = x[:, :2].long() #<--- embedding lookups require integer indices
        extra = x[:, 2:].float() #<--- (batch, n_extra) -- empty if n_extra=0

        # Look up both embeddings and concatenate with the extra features: (batch, 2*emb_dim + n_extra)
        z = torch.cat([self.stock_emb(idx[:, 0]), self.cust_emb(idx[:, 1]), extra], dim=1)

        # Map (batch, 2*emb_dim + n_extra) --> (batch, d_last) --> (batch, 1) --> (batch,) [flat vector]
        return self.head(self.trunk(z)).squeeze(1) #<--- log of predicted mean (using a log link function)

# Define a training function that fits a model by minibatch SGD with early stopping on validation loss
# Return the model with weights determined from the best validation epoch
def train(model: nn.Module, loss_fn: Callable, X_tr_t: torch.Tensor, y_tr_t: torch.Tensor, X_va_t: torch.Tensor, y_va_t: torch.Tensor, epochs: int = 100, lr: float = 1e-1, batch_size: int = 1024, patience: int = 10, lr_patience: int = 3, lr_factor: float = 0.3) -> nn.Module:
    # Create the optimizer with all learnable weights in the network
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    # Reduce the learning rate over time: lr -> lr_factor * lr after lr_patience epochs without improvement
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
            log_mu = model(xb) #<--- call EmbeddingNet.forward() to get predicted log-mean
            loss = loss_fn(log_mu, yb) #<--- compute loss
            opt.zero_grad() #<--- set current stored gradients to 0
            loss.backward() #<--- compute d/dtheta(loss) for each weight

            # Treating all gradients as a single vector, ensure it has norm <= 5
            # This caps the gradient step size, preventing a batch with large orders from destabilizing training
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            opt.step() #<--- apply update to each parameter
            train_loss += loss.item() * len(xb) #<--- update training loss with full sum over batch
        train_loss /= len(X_tr_t) #<--- compute average training loss for the epoch

        model.eval() #EVALUATION mode
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item() #<--- evaluate the validation loss

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
    model.load_state_dict(best_state)
    return model

# Define a prediction function that runs the trained model on X_t and returns predicted 'Quantity'
def predict(model: nn.Module, X_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        model.eval() #EVALUATION mode
        log_mu = model(X_t) #<--- the network output is log(predicted mean)
        mu = torch.exp(log_mu) #<--- undo the log link to get predicted 'Quantity'
        mu = mu.numpy().astype(np.float64) #<--- convert to a NumPy array for sklearn metrics
        return mu

# Train the network and assess its performance on the test set
model_embed = EmbeddingNet(len(mappings["StockCode"]), len(mappings["CustomerID"]), emb_dim=16, init_bias=float(np.log(y_train_cap.mean())))
loss_fn = nn.PoissonNLLLoss(log_input=True) #<--- Poisson loss, model outputs log-mean
model_embed = train(model_embed, loss_fn, X_tr_t, y_tr_t, X_va_t, y_va_t, epochs=150, lr=5e-2)

metrics["Embeddings"]["Train"] = summarize_result("Neural Network on Training Set", y_train, predict(model_embed, X_tr_t))
metrics["Embeddings"]["Validation"] = summarize_result("Neural Network on Validation Set", y_val, predict(model_embed, X_va_t))
metrics["Embeddings"]["Test"] = summarize_result("Neural Network on Test Set", y_test, predict(model_embed, X_te_t))
####################################################################

####################################################################
#
#                          Problem 2(d)
#
####################################################################
torch.manual_seed(SEED)

# Standardize log(target-encoded features) with training statistics, so these inputs are centered with unit scale
# Since we are initializing near-zero embeddings, this helps keep SGD stable during training
te_log_train = np.log(X_train_te[feature_cols].values)
te_mean, te_std = te_log_train.mean(axis=0), te_log_train.std(axis=0)

# Combine the embedding indices from (c) with the target-encoded features from (b) into one (N, 4) tensor:
# Columns 0-1 = StockCode/CustomerID indices, Columns 2-3 = standardized log of the target-encoded StockCode/CustomerID
# We take the log since the network predicts log(mean Quantity), so the encoded means enter on the same (log) scale as the output
def combine(X_idx_t: torch.Tensor, X_te: pd.DataFrame) -> torch.Tensor:
    te = (np.log(X_te[feature_cols].values) - te_mean) / te_std
    te = torch.tensor(te, dtype=torch.float32)
    return torch.cat([X_idx_t.float(), te], dim=1)

X_tr_c = combine(X_tr_t, X_train_te)
X_va_c = combine(X_va_t, X_val_te)
X_te_c = combine(X_te_t, X_test_te)

# Train the combined network with the same settings as (c)
model_combined = EmbeddingNet(len(mappings["StockCode"]), len(mappings["CustomerID"]), emb_dim=16, init_bias=float(np.log(y_train_cap.mean())), n_extra=2)
model_combined = train(model_combined, loss_fn, X_tr_c, y_tr_t, X_va_c, y_va_t, epochs=150, lr=5e-2)

metrics["Combined"]["Train"] = summarize_result("Combined Network on Training Set", y_train, predict(model_combined, X_tr_c))
metrics["Combined"]["Validation"] = summarize_result("Combined Network on Validation Set", y_val, predict(model_combined, X_va_c))
metrics["Combined"]["Test"] = summarize_result("Combined Network on Test Set", y_test, predict(model_combined, X_te_c))

# Save the model comparison for the report
save_results("problem2_metrics", {m: {sp: {k: float(v) for k, v in r.items()} for sp, r in d.items()} for m, d in metrics.items()})
save_table("p2_metrics_table", ["Model", "Train deviance", "Val.\\ deviance", "Test deviance", "Test MAE", "Test RMSE"], [[m, f"{d['Train']['deviance']:.2f}", f"{d['Validation']['deviance']:.2f}", f"{d['Test']['deviance']:.2f}", f"{d['Test']['MAE']:.2f}", f"{d['Test']['RMSE']:.2f}"] for m, d in metrics.items()], "lrrrrr")

# Early stopping on a noisy SGD validation loss makes a single run sensitive to the random seed (initialization and batch order)
# To check whether the embedding vs. combined comparison is robust, we refit both networks under additional seeds with the same data split
seed_runs = {m: [{"Validation": metrics[m]["Validation"]["deviance"], "Test": metrics[m]["Test"]["deviance"], "MAE": metrics[m]["Test"]["MAE"]}] for m in ("Embeddings", "Combined")}
extra_seeds = [1, 2, 3, 4]

for seed in extra_seeds:
    for m, X_tr_m, X_va_m, X_te_m, n_extra in [("Embeddings", X_tr_t, X_va_t, X_te_t, 0), ("Combined", X_tr_c, X_va_c, X_te_c, 2)]:
        torch.manual_seed(seed)
        net = EmbeddingNet(len(mappings["StockCode"]), len(mappings["CustomerID"]), emb_dim=16, init_bias=float(np.log(y_train_cap.mean())), n_extra=n_extra)
        net = train(net, loss_fn, X_tr_m, y_tr_t, X_va_m, y_va_t, epochs=150, lr=5e-2)
        res_va = summarize_result(f"{m} (seed {seed}) on Validation Set", y_val, predict(net, X_va_m))
        res_te = summarize_result(f"{m} (seed {seed}) on Test Set", y_test, predict(net, X_te_m))
        seed_runs[m].append({"Validation": res_va["deviance"], "Test": res_te["deviance"], "MAE": res_te["MAE"]})

# Summarize mean +/- standard deviation and compare the two models seed by seed (Combined - Embeddings)
def mean_sd(v: list) -> str:
    return f"${np.mean(v):.2f} \\pm {np.std(v, ddof=1):.2f}$"

diff = {k: [c[k] - e[k] for e, c in zip(seed_runs["Embeddings"], seed_runs["Combined"])] for k in ("Validation", "Test", "MAE")}
print("Test deviance by seed: ", {m: [round(r["Test"], 2) for r in runs] for m, runs in seed_runs.items()})
print("Combined - Embeddings (test deviance) by seed: ", [round(v, 2) for v in diff["Test"]])

# Save the seed comparison for the report
save_results("problem2_seed_runs", {"seeds": [SEED] + extra_seeds, **seed_runs})
save_table("p2_seeds_table", ["Model", "Val.\\ deviance", "Test deviance", "Test MAE"], [[m, mean_sd([r["Validation"] for r in runs]), mean_sd([r["Test"] for r in runs]), mean_sd([r["MAE"] for r in runs])] for m, runs in seed_runs.items()] + [["Combined $-$ Embeddings", mean_sd(diff["Validation"]), mean_sd(diff["Test"]), mean_sd(diff["MAE"])]], "lrrr")
####################################################################################################################################

####################################################################
#
#                          Problem 2(e)
#
####################################################################
# Test-set predictions from each model
test_preds = {
    "Baseline": np.full(len(y_test), y_train.mean()),
    "Target Encoding": model.predict(X_test_te),
    "Embeddings": predict(model_embed, X_te_t),
    "Combined": predict(model_combined, X_te_c),
}

# Absolute error of each model on each test row
errors = pd.DataFrame({name: np.abs(y_test.values - p) for name, p in test_preds.items()})

# Cosmetic for LaTeX
TEX_GROUP_LABELS = {"rare (<=5)": r"rare ($\le 5$)", "medium (6-50)": r"medium (6--50)", "common (>50)": r"common ($>50$)"}

# Group test rows by how many training rows their category has, and compare MAE across groups
for col in feature_cols:
    train_counts = X_train[col].value_counts().to_dict() #<--- number of training rows per category
    n_train = X_test[col].astype(object).map(train_counts).fillna(0) #<--- 0 if unseen in training
    errors["group"] = pd.cut(n_train.values, bins=[-1, 5, 50, np.inf], labels=["rare (<=5)", "medium (6-50)", "common (>50)"])

    mae_by_group = errors.groupby("group", observed=True).mean().round(2)
    group_sizes = errors["group"].value_counts().sort_index()
    print(f"\nTest MAE by {col} training frequency:")
    print(mae_by_group)
    print("Test rows per group:", group_sizes.to_dict())

    # Save the grouped MAEs for the report
    save_results(f"problem2_mae_by_{col}", {str(g): {"test_rows": int(group_sizes[g]), **{m: float(v) for m, v in row.items()}} for g, row in mae_by_group.iterrows()})
    save_table(f"p2_mae_{col.lower()}_table", ["Training frequency", "Test rows"] + list(mae_by_group.columns), [[TEX_GROUP_LABELS[str(g)], f"{group_sizes[g]:,}"] + [f"{v:.2f}" for v in row] for g, row in mae_by_group.iterrows()], "lr" + "r" * len(mae_by_group.columns))
####################################################################