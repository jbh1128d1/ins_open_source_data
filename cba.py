# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:59:08 2021

@author: jordan.howell
"""


def cba(bp, mf, prem, lr, cs):
    bp = bp
    mf = mf
    tg = .95
    oe = tg - bp - mf
    
    el = prem * lr
    mfd = prem * mf
    oed = prem * oe
    exchange_gain = prem - mfd - oed - el
    
    fgi_gain = mfd + cs
    
    enterprise_gain = exchange_gain + fgi_gain
    
    print("Expected Loss Dollars: " ,el, "\n",
          'Management Fee Dollars: ', mfd, "\n",
          'Other Expense Dollars: ', oed, "\n")
    
    print("Exchange P&L: ", exchange_gain, "\n",
          "FGI P&L: ", fgi_gain, "\n",
          "Enterprise P&L: ", enterprise_gain)