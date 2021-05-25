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

def get_results_comparison_charts(df, old_model, new_model, eu, earned_premium, incurred, title):
    '''
    df = test model set
    old_model = current model variable
    new_model = new model variable 
    eu = variable of earned units
    earned_premium = variable of earned premium
    incurred = variable of incurred costs
    title = The start of each title
    '''
    @njit
    def partition(c, n):
        delta = c[-1] / n
        group = 1
        indices = [group]
        total = delta

        for left, right in zip(c, c[1:]):
            left_diff = total - left
            right_diff = total - right
            if right > total and abs(total - right) > abs(total - left):
                group += 1
                total += delta
            indices.append(group)

        return indices
    
    
    def test_results(df, eu, model_pred, comparison_models, incurred, title):
        '''
        Makes Lift Chart for DOIs

        df = data frame of the data used

        eu = coverage-specific earned units

        model_pred = the model the user wants to sort the values by

        comparison_models = list of models to compare to include the model_pred model

        incurred = coverage specific incurred

        title = title of the charts

        '''
        
        df = df.loc[df[eu] > 0].copy()
        
        df = df.sort_values(model_pred, ascending = True)

        df['result'] = partition(df[eu].values.cumsum(), n = 10)

        top_group_df = pd.DataFrame() 

        bottom_group_df = pd.DataFrame()

        final_model_df = pd.DataFrame()

        for model in comparison_models:
            df['pred_model4'] = df[model] * df[eu]

            top_group = df.groupby('result')[['pred_model4', incurred, eu]].agg('sum')
            top_group['decile_model4_pp'] = top_group['pred_model4']/top_group[eu]
            top_group['decile_act_pp'] = top_group[incurred]/top_group[eu]
            top_group['model'] = model

            top_group_df = top_group_df.append(top_group)

            bottom_group = df[['pred_model4', incurred, eu]].agg('sum')
            bottom_group['model4_pp_total'] = bottom_group['pred_model4']/bottom_group[eu]
            bottom_group['act_pp_total'] = bottom_group[incurred]/bottom_group[eu]
            bottom_group['norm_model4'] = bottom_group['act_pp_total']/bottom_group['model4_pp_total']
            bottom_group['model'] = model

            bottom_group_df = bottom_group_df.append(bottom_group, ignore_index=True)

            final_model = pd.DataFrame(top_group['decile_model4_pp']*bottom_group['norm_model4'])
            final_model['actual_pp_norm'] = pd.DataFrame(top_group['decile_act_pp'])
            final_model = final_model.reset_index()
            final_model.columns = ['result', model, 'Actual']
            final_model = pd.melt(final_model, id_vars = ['result'], value_vars = list(final_model.columns)[1:], var_name = 'Model'
                , value_name = 'Pure Premium')
            #final_model['model'] = model

            final_model_df = final_model_df.append(final_model, ignore_index=True)

        final_model_df.drop_duplicates(inplace = True)

        plt.rcParams['figure.figsize'] = [16,10]
        plt.rcParams['figure.dpi'] = 150
        ax = plt.figure(facecolor='w', edgecolor='k')
        ax = sns.pointplot(x ='result', y = 'Pure Premium', data = final_model_df, color = 'red', hue = 'Model')
        ax = plt.title(title)
        ax = plt.ylabel('Pure Premium')
        ax = plt.xlabel('Decile')
        ax = plt.xticks(rotation='vertical', fontsize = 8)
        
        return ax, final_model_df
    
    def loss_ratio_chart(df, eu, model_pred, incurred, earned_premium, title):
    
        '''
        Makes Lift Chart for DOIs

        df = data frame of the data used

        eu = coverage-specific earned units

        model_pred = the model the user wants to sort the values by

        comparison_models = list of models to compare to include the model_pred model

        incurred = coverage specific incurred

        earned_premium = earned premium for cover

        title = title of the charts

        '''
        df = df.loc[df[eu] > 0].copy()

        df = df.sort_values(model_pred, ascending = True)

        df['result'] = partition(df[eu].values.cumsum(), n = 10)

        top_group = df.groupby('result')[[incurred, eu, earned_premium]].agg('sum').reset_index()

        top_group['loss_ratio'] = round((top_group[incurred]/top_group[earned_premium])*100, 2)

        top_group['model'] = model_pred

        plt.rcParams['figure.figsize'] = [16,10]
        plt.rcParams['figure.dpi'] = 150
        ax = plt.figure(facecolor='w', edgecolor='k')
        ax = sns.barplot(x ='result', y = 'loss_ratio', data = top_group, color = 'deepskyblue')
        ax = plt.title(title)
        ax = plt.ylabel('Loss Ratio')
        ax = plt.xlabel('Decile')
        ax = plt.xticks(rotation='vertical', fontsize = 8)


        return ax, top_group
    
    #calculate GINI and plot lorenz curve
    
    def gini_graph(df, new_model, eu):

        gini_test = df.sort_values(new_model)
        gini_test['cumulative_prediction'] = gini_test[new_model].cumsum()
        gini_test['cumulative_eu'] = gini_test[eu].cumsum()
        gini_test['cumulative_percentage_loss'] = gini_test['cumulative_prediction']/gini_test[new_model].sum()
        gini_test['cumulative_percentage_exposure'] = gini_test['cumulative_eu']/gini_test[eu].sum() 

        gini_test['quantile'] = partition(gini_test[eu].values.cumsum(), n = 5)
        quant = pd.DataFrame(gini_test.groupby('quantile')[new_model].sum())
        quant.columns = ['loss']
        quant['cumulative_loss'] = quant['loss'].cumsum()
        quant['cum_perccentage']  = quant['loss']/quant['loss'].sum()

        y = np.array(quant['cum_perccentage'])
        y_pe = np.linspace(0,1,len(y))

        # Compute the area using the composite trapezoidal rule.
        area_lorenz = np.trapz(y, dx=5)

        # Calculate the area below the perfect equality line.
        area_perfect = np.trapz(y_pe, dx=5)

        # Seems to work fine until here. 
        # Manually calculated Gini using the values given for the areas above 
        # turns out at .58 which seems reasonable?

        Gini = (area_perfect - area_lorenz)/area_perfect
        
        plt.rcParams['figure.figsize'] = [16,10]
        plt.rcParams['figure.dpi'] = 150
        ax = plt.figure(facecolor='w', edgecolor='k')
        ax = sns.lineplot(x = 'cumulative_percentage_exposure', y = 'cumulative_percentage_loss', data = gini_test, label = 'Observed')
        ax = sns.lineplot(x = 'cumulative_percentage_exposure', y = 'cumulative_percentage_exposure', data = gini_test, label = 'Equality')
        ax = plt.xlabel('Exposure Percentage')
        ax = plt.ylabel('Loss Percentage')
        ax = plt.title('Gini Index and the Lorenz Curve -'+ title + 'GINI = '+str(round(Gini, 3)))

        return ax
 
        
    #Single lift Chart of the new model
    #df = df.sort_values(new_model, ascending = False)
    test_results(df = df, eu = eu, model_pred = new_model, comparison_models = [new_model], incurred = incurred
                    ,title =  title+'New Model Lift Chart')
    
    #Single lift Chart of the old model
    #df = df.sort_values(old_model, ascending = False)
    test_results(df = df, eu = eu, model_pred = old_model, comparison_models = [old_model], incurred = incurred
                    ,title =  title+'Old Model Lift Chart')
    
    #Loss Ratio Chart New Model
    #df = df.sort_values(new_model, ascending = False)
    loss_ratio_chart(df = df, eu = eu, model_pred = new_model, incurred = incurred
             , earned_premium = earned_premium ,title =  title+'New Model Loss Ratio')
    
    #Loss Ratio Chart Old Model
    #df = df.sort_values(old_model, ascending = False)
    loss_ratio_chart(df = df, eu = eu, model_pred = old_model, incurred = incurred
             , earned_premium = earned_premium ,title =  title+'Old Model Loss Ratio')
    
    #Produce Double lift Chart
    
    df['sort_order'] = df[old_model]/df[new_model]
    
    #df = df.sort_values('sort_order', ascending = False)

    test_results(df = df, eu = eu, model_pred = 'sort_order'
                    , comparison_models = [old_model, new_model], incurred = incurred
                    ,title =  title+'Double Lift Chart')
    

    gini_graph(df, new_model, eu)