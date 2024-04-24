import pandas as pd
from scipy import stats
import numpy as np

# Load data
grid_results = pd.read_csv(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\grid_search_output_results\wine_results_grid.csv")
optuna_results = pd.read_csv(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\optuna_results\wine_optuna_results.csv")


# Sample extraction assuming 'mse' column holds the results
mse_grid = grid_results['mse']
mse_optuna = optuna_results['MSE']

# Normality Test using Shapiro-Wilk
normality_grid = stats.shapiro(mse_grid)
normality_optuna = stats.shapiro(mse_optuna)

print(f"Grid Search Normality: p-value = {normality_grid.pvalue}")
print(f"Optuna Normality: p-value = {normality_optuna.pvalue}")

# Homogeneity of variances using Levene's Test
levene_test = stats.levene(mse_grid, mse_optuna)
print(f"Levene test for equal variances: p-value = {levene_test.pvalue}")

# Choose test based on normality and homogeneity
if normality_grid.pvalue > 0.05 and normality_optuna.pvalue > 0.05 and levene_test.pvalue > 0.05:
    # Perform ANOVA
    f_val, p_val = stats.f_oneway(mse_grid, mse_optuna)
    print(f"ANOVA F-test: F-value = {f_val}, p-value = {p_val}")
    
    # Calculate Eta Squared
    eta_squared = (f_val * len(mse_grid)) / (f_val * len(mse_grid) + len(mse_grid))
    print(f"Eta Squared = {eta_squared}")

    # Cohen's d
    cohen_d = (np.mean(mse_grid) - np.mean(mse_optuna)) / np.sqrt((np.std(mse_grid) ** 2 + np.std(mse_optuna) ** 2) / 2)
    print(f"Cohen's d = {cohen_d}")
else:
    # Perform Kruskal-Wallis Test
    kw_val, p_val = stats.kruskal(mse_grid, mse_optuna)
    print(f"Kruskal-Wallis test: H-statistic = {kw_val}, p-value = {p_val}")
    
    # Cohen's d (not typically used with Kruskal-Wallis but showing as example)
    cohen_d = (np.mean(mse_grid) - np.mean(mse_optuna)) / np.sqrt((np.std(mse_grid) ** 2 + np.std(mse_optuna) ** 2) / 2)
    print(f"Cohen's d = {cohen_d}")
