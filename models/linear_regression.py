import pandas as pd
import os
import matplotlib.pyplot as plt
from preprocessing import input_data_to_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

X_train, y_train, X_test, y_test, features = input_data_to_model()

# Model Training

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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
    os.path.join("result_modeling", "LR PvA Scatter.png"),
    dpi=400,
    bbox_inches="tight",
)
plt.close()

with open("model_errs_and_coefs/lin_reg_errors.txt", "w") as fout:
    fout.write("The r2 score is " + str(round(r2, 3)) + "\n")
    fout.write("The MSE is " + str(round(mse, 3)) + "\n")
    fout.write("The RMSE is " + str(round(rmse, 3)) + "\n")

coefficients = model.coef_
coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefficients})
coef_df["Abs_Value_Coefs"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(by="Abs_Value_Coefs", ascending=False)
coef_df.to_csv("model_errs_and_coefs/lin_reg_coefs.csv")
