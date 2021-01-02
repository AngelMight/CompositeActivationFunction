# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:15:32 2021

@author: kstoy
"""

import pandas
df = pandas.read_csv('train.csv', index_col='Id')
print(df)