## Title
Deconstructing Household Energy Use: A Machine Learning Approach

## Description (Abstract)
Understanding residential energy consumption is crucial not only for developing effective energy policies but also for mitigating the environmental impact of energy generation which is a major contributor of greenhouse gases. This paper identifies the core determinants influencing U.S. household energy consumption in 2020 and how they changed from previous years. Several machine learning algorithms including Random Forest, Gradient Boost, and Polynomial Regression, are explored to determine the best model to forecast energy consumption, using features from a detailed dataset of various residential characteristics published in the 2020 Residential Energy Consumption Survey (RECS). The models were assessed for accuracy using standard performance metrics. Based on this investigation, Gradient Boost emerged as the best performing model explaining 91.1% of the variability in the energy consumption rates. Using this optimized model for 2020, the analysis identified the cost of natural gas, the cost of electricity, heating degree days (a measure of heating demand), and the size of the residence as the primary factors driving household energy consumption. These determinants differ from those in 2015, highlighting the growing sensitivity of U.S. households to energy prices. This work is important as a quantitative understanding of the drivers of residential energy consumption that can empower the development and evaluation of sustainable energy practices and policies, improving energy efficiency and reducing greenhouse gas emissions.

## Attribution
This project was completely developed by me, Aditya Ramanathan.

## Date
February 12th, 2025

## Setup and Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the data_simplification.py (must be run first)
4. For data_analysis, cd into the data_analysis directory and run eda_recs.py
5. For modeling, enter the models directory and run preprocessing.py. Then any of the _____.py files can be run.

## Code Tree

```
├── README.md
├── data_analysis
│   ├── eda_recs.py
│   ├── recs_filtered_2020_eda.csv
│   ├── recs_filtered_2020_eda_with_all_energies.csv
│   └── statistics.csv
├── data_simplification.py
├── models
│   ├── cubic_regression.py
│   ├── elasticnet_regression.py
│   ├── gradient_boost.py
│   ├── hist_gradient_boost.py
│   ├── linear_regression.py
│   ├── model_errs_and_coefs
│   │   ├── Gradient Boost.png
│   │   ├── Hist Gradient Boost.png
│   │   ├── Random Forest.png
│   │   ├── cubic_reg_coefs.csv
│   │   ├── cubic_reg_errors.txt
│   │   ├── elastic_net_coefs.csv
│   │   ├── elastic_net_errors.txt
│   │   ├── error_table.csv
│   │   ├── error_table.py
│   │   ├── gb_plot_df.csv
│   │   ├── gradient_boost_errors.txt
│   │   ├── hgb_plot_info.json
│   │   ├── hist_gradient_boost_errors.txt
│   │   ├── lin_reg_coefs.csv
│   │   ├── lin_reg_errors.txt
│   │   ├── quad_reg_coefs.csv
│   │   ├── quad_reg_errors.txt
│   │   ├── random_forest_errors.txt
│   │   ├── rf_plot_df.csv
│   │   └-─ tbm_result_plotting.py
│   ├── preprocessing.py
│   ├── quadratic_regression.py
│   ├── random_forest.py
│   └-─ recs_filtered_2020.csv
│       ─└ best_results_1736562186.313877_random_forest.json
└-- recs_metadata_2020.csv
```
