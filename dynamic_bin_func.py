# -*- coding: utf-8 -*-
"""
Created on Mon May 24 09:27:04 2021

@author: jordan.howell
"""

import pandas as pd
import numpy as np


def dynamic_bin_func(df, columns, weight, minimum=1000):
    """
    Parameters
    ----------
    df : dataframe
    column : column to be binned
    weight : column that will dictate the bin
    minimum : minimum weight per bin
    
    Returns
    -------
    df : dataframe with new binned column
    
    """
    for column in columns:
        bins = [-np.inf]
        labels = [] 
        hold_over = []
        for i in sorted(df[column].unique()):
            g = df[df[column] == i].groupby(column).agg({weight:'sum'}).reset_index()
    
            if g[weight].values[0] < minimum:
                if hold_over is None:
                    hold_over.append(g[weight].values[0])
        
                elif (sum(hold_over) + g[weight].values[0]) < minimum:
                    hold_over.append(g[weight].values[0])
                 
                    
                elif (sum(hold_over) + g[weight].values[0]) >= minimum:
                    hold_over.clear()
                    bins.append(g[column].values[0])
                    labels.append(g[column].values[0])
        
            elif g[weight].values[0] >= minimum:
                bins.append(g[column].values[0])
                labels.append(g[column].values[0])
    
        bins.pop()
        bins.append(np.inf)
    
    
        str_column = str(column)+str("_binned")
        # print(str_column)
        df[str_column] = pd.cut(df[column],
                        bins = bins,
                        labels = labels)

    return df
    
    
    
            

            

