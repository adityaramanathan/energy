import pandas as pd


def extract(file_name):
    numbers = []

    with open(file_name, "r") as fin:
        lines = fin.readlines()

    for line in lines:
        val = 0
        ind = line.index("is")
        val = float(line[ind + 2 :].strip())
        numbers.append(val)

    return numbers


errs_lin = extract("lin_reg_errors.txt")
errs_quadratic = extract("quad_reg_errors.txt")
errs_cubic = extract("cubic_reg_errors.txt")
errs_el = extract("elastic_net_errors.txt")
errs_rf = extract("random_forest_errors.txt")
errs_gb = extract("gradient_boost_errors.txt")
errs_hgb = extract("hist_gradient_boost_errors.txt")

errs = [errs_lin, errs_quadratic, errs_cubic, errs_el, errs_rf, errs_gb, errs_hgb]
columns = [
    "linear regression",
    "quadratic regression",
    "cubic regression",
    "elastic net",
    "random forest",
    "gradient boost",
    "histogram gradient boost",
]

df = pd.DataFrame(list(zip(*errs)), columns=columns)
pd.set_option("display.float_format", "{:.10f}".format)
df.to_csv("error_table.csv")
