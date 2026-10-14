"""Problem 5 (classification)"""
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import save_results, save_table

# Load the data and randomly split into 80% train, 20% test
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

# First, we perform a basic logistic regression
baseline = make_pipeline(StandardScaler(), LogisticRegression())
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
print(f"sklearn Baseline Test Accuracy: {baseline_acc:.4f}")

# (a) Now we do it with autodifferentiation
# Start by borrowing the scaled train/test data from the pipeline and converting NumPy arrays into PyTorch tensors
scaler = baseline[0]
X_train_t = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
X_test_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
n_train, n_features = X_train_t.shape #[0] = training rows, [1] = number of features

# Define the logistic regression model explicitly
def logistic_model(X, w, b):
	# matmul(X, w) = (Xw)_i = x_ij w_j = w^T x_i
	return torch.sigmoid(torch.matmul(X, w) + b)

# Define the loss function (binary cross-entropy loss)
def bce_loss(f, y, eps=1e-7):
	# Clip the output of the input function to avoid numerical instabilities
	f = torch.clamp(f, eps, 1 - eps)
	
	# Return the mean (over all observations) so that we have a singular quantity to minimize
	return -(y * torch.log(f) + (1 - y) * torch.log(1 - f)).mean()

# This training function performs the stochastic gradient descent based on a set of inputs
def train_sgd(model, lr=0.1, n_epochs=100, batch_size=32):
	torch.manual_seed(7)
	w = torch.zeros(n_features, requires_grad=True)
	b = torch.zeros(1, requires_grad=True)
	
	# Set up SGD with learning rate set by *lr*
	optimizer = torch.optim.SGD([w, b], lr=lr)
	
	# Loop over epochs
	for epoch in range(n_epochs):
		# Generate random ordering each epoch to shuffle data
		perm = torch.randperm(n_train)
	
		# Loop through observations *batch_size* at a time
		for start in range(0, n_train, batch_size):
			idx = perm[start : start + batch_size]
			# Pass *idx* observations through model and compute loss
			loss = bce_loss(model(X_train_t[idx], w, b), y_train_t[idx])
		
			optimizer.zero_grad() #<--- clear gradients from last step
			loss.backward() #<--- work backward through operations and compute+store dL/dw, dL/db
			optimizer.step() #<--- apply the SGD updates: w -= lr * dL/dw, b -= lr * dL/db
	
		# Print diagnostics every 10 epochs, or on the last epoch
		if epoch % 10 == 0 or epoch == n_epochs - 1:
			with torch.no_grad():
				full_loss = bce_loss(model(X_train_t, w, b), y_train_t).item()
			print(f"  epoch {epoch:3d}  train loss {full_loss:.4f}")
	return w.detach(), b.detach()

# Function to determine accuracy of model when applied to test data
def test_accuracy(model, w, b):
	with torch.no_grad():
		# For binary categorization, convert test probability to binary value 0/1
		pred = (model(X_test_t, w, b) >= 0.5).float()
	
	accuracy = (pred == y_test_t).float().mean().item()
	return accuracy
	
# (b) The only difference is we need to implement a probit regression model
NORMAL = torch.distributions.Normal(0.0, 1.0)
def probit_model(X, w, b):
	return NORMAL.cdf(torch.matmul(X,w) + b)

# Now, actually train the models
print("(a) Logistic regression:")
w_log, b_log = train_sgd(logistic_model, n_epochs=200)
print("(b) Probit regression:")
w_pro, b_pro = train_sgd(probit_model, n_epochs=200)

accs = {
	"sklearn baseline": baseline_acc,
	"SGD (logistic model)": test_accuracy(logistic_model, w_log, b_log),
	 "SGD (probit model)": test_accuracy(probit_model, w_pro, b_pro),
}

# Save results
n_test = len(y_test)
print(f"{'Method':<22}{'Accuracy':>10}{'Correct':>12}")
rows = []
for name, acc in accs.items():
	n_correct = round(acc * n_test)
	print(f"{name:<22}{acc:>10.2%}{f'{n_correct}/{n_test}':>12}")
	rows.append([name, f"{100 * acc:.2f}\\%", f"{n_correct}/{n_test}"])

save_results("problem5_classification", {k: {"test_accuracy": v} for k, v in accs.items()})
save_table("p5_class_table", ["Method", "Test accuracy", "Correct"], rows, "lrr")
