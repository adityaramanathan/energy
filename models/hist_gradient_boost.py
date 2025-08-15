import os
import matplotlib.pyplot as plt
from preprocessing import input_data_to_model
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import math
import json
from datetime import datetime

X_train, y_train, X_test, y_test, features = input_data_to_model()

# Model Training

param_grid = {
    "max_iter": [100, 200, 300],
    "learning_rate": [0.05, 0.075, 0.1, 0.15, 0.3],
    "max_depth": [None, 3, 4, 5, 6],
    "l2_regularization": [0.1, 0.2, 0.9, 1.0],
    "min_samples_leaf": [9, 10, 11],
}

model = HistGradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="r2",  # Use R^2 scoring
    n_jobs=-1,
    cv=10,
    verbose=1,
)

grid_search.fit(X_train, y_train)

estimator = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

timestamp = datetime.now().timestamp()

json_path = os.path.join(
    "results_histgb", "best_results_" + str(timestamp) + "_hist_gradient_boost.json"
)
with open(json_path, "w") as f:
    json.dump({"best_params": best_params, "best_score": best_score}, f)

y_pred = estimator.predict(X_test)

# Error Results

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Predicted vs. Actual Plot

plt.figure(figsize=(24, 18))
plt.plot(y_pred, y_test, marker="o", label="Point", linestyle="")
plt.plot(y_test, y_test, label="Line", color="black", linestyle="-")
plt.xlabel("Predicted", fontsize=24)
plt.xticks(fontsize=18)
plt.ylabel("Actual", fontsize=24)
plt.yticks(fontsize=18)
plt.title("Predicted vs. Actual", fontsize=30)
plt.legend()
plt.savefig(
    os.path.join("result_modeling", "HGB PvA Scatter.png"),
    dpi=400,
    bbox_inches="tight",
)
plt.close()

with open("model_errs_and_coefs/hist_gradient_boost_errors.txt", "w") as fout:
    fout.write("The r2 score is " + str(round(r2, 3)) + "\n")
    fout.write("The MSE is " + str(round(mse, 3)) + "\n")
    fout.write("The RMSE is " + str(round(rmse, 3)) + "\n")

# Coefficient Results

results = permutation_importance(estimator, X_test, y_test, scoring="r2")
importances = results.importances_mean

plot_data = {"features": features, "importances": importances.tolist()}

plot_data_path = os.path.join("model_errs_and_coefs", "hgb_plot_info.json")
with open(plot_data_path, "w") as f:
    json.dump(plot_data, f)
