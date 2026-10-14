"""Problem 4"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from common import FIGURES_DIR, load_ale, save_results, save_table

X, y = load_ale()

#Random data split into 75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
print("Train:", X_train.shape, " Test:", X_test.shape)

#We will use the recommended models from the problem
models = {
	"Linear": LinearRegression(),
	"SVR with RBF": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.05)),
	"Decision tree": DecisionTreeRegressor(max_depth=4, random_state=99),
	"Neural net": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, random_state=99)),
}

residuals = {} 
results = {}

# Loop through models --> fit test data --> compute residuals and perform normaltest
for name, model in models.items():
	model.fit(X_train, y_train)
	res = y_test - model.predict(X_test)
	residuals[name] = res
	stat, p = stats.normaltest(res)
	results[name] = {
		"test_rmse": float(np.sqrt(np.mean(res**2))),
		"normaltest_stat": float(stat),
		"normaltest_p": float(p),
		"reject_normality_0.05": bool(p < 0.05),
	}

# Make qqplot figures
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
for ax, (name, res) in zip(axes.ravel(), residuals.items()):
	stats.probplot(res, dist="norm", plot=ax)
	ax.set_title(f"QQ plot: {name}")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "p4_qqplots.png", dpi=200)
plt.close(fig)

# Output results of model testing
print(f"{'Model':15s} {'Test RMSE':>10s} {'Statistic':>10s} {'p-value':>10s} Reject normality?")
rows = []
for name, r in results.items():
	verdict = "Yes" if r["reject_normality_0.05"] else "No"
	print(f"{name:15s} {r['test_rmse']:10.3f} {r['normaltest_stat']:10.3f} {r['normaltest_p']:10.3f} {verdict}")
	rows.append([name, f"{r['test_rmse']:.3f}", f"{r['normaltest_stat']:.3f}", f"{r['normaltest_p']:.3f}", verdict])

save_results("problem4", results)
save_table("p4_table", ["Model", "Test RMSE", "Statistic", "$p$-value", "Reject at 0.05?"], rows, "lrrrc")
