# -*- coding: utf-8 -*-
"""
Created on Mon May 24 08:36:07 2021

@author: jordan.howell
"""

import pandas as pd
import numpy as np
import ins_func
from dynamic_bin_func import dynamic_bin_func as dbf
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor, GammaRegressor
import ins_func as isf
import matplotlib.pyplot as plt
import patsy
from get_results_comparison_charts import get_results_comparison_charts

pd.options.display.max_columns = 1500
pd.options.display.max_rows = 1500

df = ins_func.load_mtpl2()

bin_columns = ["VehAge", "DrivAge"]
cat_columns =  ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
log_columns = ["Density"]

df = isf.df_model(df = df, cat_cols = cat_columns,
                  log_calls = log_columns, bin_calls = bin_columns, 
                  weight = 'Exposure')





df["Frequency"] = df["ClaimNb"] / df["Exposure"]



df.columns = df.columns.str.replace(".0","")
words = df.columns

string_formula = ""

for word in words:
    string_formula += word
    string_formula += " + "
    
string_formula

formula1 = "Frequency ~ VehAge_binned + DrivAge_binned + VehBrand_B1 + VehBrand_B + VehBrand_B11 + VehBrand_B12 + VehBrand_B13 + VehBrand_B14 + VehBrand_B2 + VehBrand_B3 + VehBrand_B4 + VehBrand_B5 + VehBrand_B6 + VehPower_4 + VehPower_5 + VehPower_6 + VehPower_7 + VehPower_8 + VehPower_9 + VehPower_ + VehPower_11 + VehPower_12 + VehPower_13 + VehPower_14 + VehPower_15 + VehGas_Diesel + VehGas_Regular + Region_R11 + Region_R21 + Region_R22 + Region_R23 + Region_R24 + Region_R25 + Region_R26 + Region_R31 + Region_R41 + Region_R42 + Region_R43 + Region_R52 + Region_R53 + Region_R54 + Region_R72 + Region_R73 + Region_R74 + Region_R82 + Region_R83 + Region_R91 + Region_R93 + Region_R94 + Area_A + Area_B + Area_C + Area_D + Area_E + Area_F + Density_log"

y, x = patsy.dmatrices(formula1, df_train, return_type = 'dataframe')
y_test, x_test = patsy.dmatrices(formula1, df_test, return_type = 'dataframe')

df_train, df_test = train_test_split(df, test_size = .20, random_state = 101)

glm_freq = PoissonRegressor()
glm_freq.fit(x, y,
             sample_weight=df_train["Exposure"])

scores = isf.score_estimator(
    glm_freq,
    x,
    x_test,
    df_train,
    df_test,
    target="Frequency",
    weights="Exposure",
)

preds =  glm_freq.predict(x_test)
df_test['preds1'] = preds
df_test = df_test.sort_values('preds1')
df_test['decile'] = isf.partition(df_test['preds1'].cumsum().values, 10)

df_test.groupby('decile').agg({'preds1':'sum', 'Frequency':'sum', 
                               'Exposure':'sum'})

df_test['baseline'] = df_test['ClaimNb'].sum()/df_test['Exposure'].sum()

get_results_comparison_charts(df = df_test, old_model = 'baseline'
                              , new_model = 'preds1', eu = 'Exposure', 
                              earned_premium = 'BonusMalus', 
                              incurred = 'ClaimAmount', title='Toy Insurance Exhibit')

# =============================================================================
# fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
# fig.subplots_adjust(hspace=0.3, wspace=0.2)
# 
# isf.plot_obs_pred(
#     df=df_train,
#     feature="DrivAge",
#     weight="Exposure",
#     observed="Frequency",
#     predicted=glm_freq.predict(X_train),
#     y_label="Claim Frequency",
#     title="train data",
#     ax=ax[0, 0],
# )
# 
# isf.plot_obs_pred(
#     df=df_test,
#     feature="DrivAge",
#     weight="Exposure",
#     observed="Frequency",
#     predicted=glm_freq.predict(X_test),
#     y_label="Claim Frequency",
#     title="test data",
#     ax=ax[0, 1],
#     fill_legend=True
# )
# 
# isf.plot_obs_pred(
#     df=df_test,
#     feature="VehAge",
#     weight="Exposure",
#     observed="Frequency",
#     predicted=glm_freq.predict(X_test),
#     y_label="Claim Frequency",
#     title="test data",
#     ax=ax[1, 0],
#     fill_legend=True
# )
# 
# isf.plot_obs_pred(
#     df=df_test,
#     feature="BonusMalus",
#     weight="Exposure",
#     observed="Frequency",
#     predicted=glm_freq.predict(X_test),
#     y_label="Claim Frequency",
#     title="test data",
#     ax=ax[1, 1],
#     fill_legend=True
# )
# =============================================================================
