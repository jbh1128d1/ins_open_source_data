from numba import njit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.api import abline_plot
import patsy
import get_results_comparison_charts
plt.style.use('fivethirtyeight')

def predict_and_compare(df, model, formula, model_preds, old_model_predictions, eu, ep,claims, incrd, title, model_set):
    '''
    df = test dataframe
    model = model to predict and compare off of
    model_preds = name of column to place model predictions in
    old_model_predictions = column of old model predictions
    eu = earned units columns
    ep = column of earned premium
    incrd = column of incurred
    title = title
    model_set = name of test data frame
    '''
    print(f"Creating test matrix....")
    y_test, x_test = patsy.dmatrices(formula, df, return_type = 'dataframe')
    #model_set_dict.update({model_set+'_x': x, model_set+'_y': y})
    print(f"Predicting results off of test data....")
    predictions = model.predict(x_test)
    df.loc[:, model_preds] = predictions
    print(f"getting results...")
    get_results_comparison_charts.get_results_comparison_charts(df = df, old_model = old_model_predictions, new_model = model_preds, eu = eu
                                  , earned_premium = ep, incurred = incrd, title = title)