import pandas as pd
import os
import matplotlib.pyplot as plt
from preprocessing import input_data_to_model
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import math
import json
from datetime import datetime

X_train, y_train, X_test, y_test, features = input_data_to_model()

# Model Training

param_grid = {
    "alpha": [0.01, 0.05, 0.1, 0.5, 1],
    "l1_ratio": [0.1, 0.8, 0.85, 0.9, 0.95],
    "max_iter": [500, 1000, 1500],  # Number of iterations
}

model = ElasticNet(random_state=42)

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
    "results_elasticnet", "best_results_" + str(timestamp) + "_elastic_net.json"
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
    os.path.join("result_modeling", "ENR PvA Scatter.png"),
    dpi=400,
    bbox_inches="tight",
)
plt.close()

with open("model_errs_and_coefs/elastic_net_errors.txt", "w") as fout:
    fout.write("The r2 score is " + str(round(r2, 3)) + "\n")
    fout.write("The MSE is " + str(round(mse, 3)) + "\n")
    fout.write("The RMSE is " + str(round(rmse, 3)) + "\n")

# Coefficient Results

coefficients = estimator.coef_
coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefficients})
coef_df["Abs_Value_Coefs"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(by="Abs_Value_Coefs", ascending=False)
coef_df.to_csv("model_errs_and_coefs/elastic_net_coefs.csv")
