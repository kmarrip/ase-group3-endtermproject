import pandas as pd
from scipy import stats
import numpy as np
from num import NUM
from sym import SYM
from statistics import mode

# Load data
grid_results = pd.read_csv("./grid_search_output_results/wine_results_grid.csv")
optuna_results = pd.read_csv("./optuna_results\wine_optuna_results.csv")

# Sample extraction assuming 'mse' column holds the results
mse_grid = grid_results["mse"]
mse_optuna = optuna_results["MSE"]

# Normality Test using Shapiro-Wilk
normality_grid = stats.shapiro(mse_grid)
normality_optuna = stats.shapiro(mse_optuna)

print(f"Grid Search Normality: p-value = {normality_grid.pvalue}")
print(f"Optuna Normality: p-value = {normality_optuna.pvalue}")

# Homogeneity of variances using Levene's Test
levene_test = stats.levene(mse_grid, mse_optuna)
print(f"Levene test for equal variances: p-value = {levene_test.pvalue}")

# Choose test based on normality and homogeneity
if (
    normality_grid.pvalue > 0.05
    and normality_optuna.pvalue > 0.05
    and levene_test.pvalue > 0.05
):
    # Perform ANOVA
    f_val, p_val = stats.f_oneway(mse_grid, mse_optuna)
    print(f"ANOVA F-test: F-value = {f_val}, p-value = {p_val}")

    # Calculate Eta Squared
    eta_squared = (f_val * len(mse_grid)) / (f_val * len(mse_grid) + len(mse_grid))
    print(f"Eta Squared = {eta_squared}")

    # Cohen's d
    cohen_d = (np.mean(mse_grid) - np.mean(mse_optuna)) / np.sqrt(
        (np.std(mse_grid) ** 2 + np.std(mse_optuna) ** 2) / 2
    )
    print(f"Cohen's d = {cohen_d}")
else:
    # Perform Kruskal-Wallis Test
    kw_val, p_val = stats.kruskal(mse_grid, mse_optuna)
    print(f"Kruskal-Wallis test: H-statistic = {kw_val}, p-value = {p_val}")

    # Cohen's d (not typically used with Kruskal-Wallis but showing as example)
    cohen_d = (np.mean(mse_grid) - np.mean(mse_optuna)) / np.sqrt(
        (np.std(mse_grid) ** 2 + np.std(mse_optuna) ** 2) / 2
    )
    print(f"Cohen's d = {cohen_d}")

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
def testNUMmid():
    num = NUM()
    toBeAdded = [2, 3, 4, 5, 6]
    for nums in toBeAdded:
        num.add(nums)

    finalMean = 0

    for val in toBeAdded:
        finalMean += val
    finalMean /= len(toBeAdded)
    assert num.mid() == finalMean


def testLow():
    num = NUM()
    toBeAdded = [1, 2, 3, 4, 99]
    for val in toBeAdded:
        num.add(val)
    assert num.lo == min(toBeAdded)


def testWithEmpty():
    sym = SYM()
    res = sym.div()
    assert res == 0


def testWithManyvalues():
    sym = SYM()
    sym.add("1")
    sym.add("2")
    sym.add("3")
    res = sym.div()
    assert res > 0


def test_sym_mid():
    sym = SYM()
    vals = [1, 1, 1, 1, 1, 1, 1, 199, 99, 9, 99, 99, 9, 99]
    for val in vals:
        sym.add(val)
    mid = mode(vals)
    assert sym.mid() == mid

    sym = SYM()
    vals = [22, 2, 2]
    for val in vals:
        sym.add(val)
    mid = mode(vals)
    assert sym.mid() == mid


testNUMmid()
testLow()
testWithEmpty()
testWithManyvalues()
