# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:31:51 2022

@author: 215-01
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.stats.multicomp import MultiComparison
import scipy.stats
from sklearn.preprocessing import RobustScaler
from pandasql import *
import statsmodels.api as sm
#%%
import subprocess
filename = r'C:\Users\215-01\Desktop\빅콘\0918_빅콘 고객유형 대출 예측.py'
dest = r'C:\Users\215-01\Desktop\빅콘\0918_빅콘 고객유형 대출 예측.ipynb'
subprocess.run(['ipynb-py-convert', filename, dest])
# In[1]:
# 경로지정
os.chdir('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908')
os.getcwd()
# In[3]:
#숫자 지수표현없이 출력
pd.options.display.float_format = '{:.10f}'.format
# In[3]:
# 플랏 한글
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
# In[4]:
loan = pd.read_csv('loan_result.csv')
log = pd.read_csv('log_data.csv')
user = pd.read_csv('user_spec.csv',encoding=('UTF-8'))
df = pd.read_csv('user_loan.csv')
#%%
print(df.isnull().sum())
#%%
df_copy = df.copy()
df_copy.dropna(subset=['employment_type'],inplace = True)
print(df_copy.isnull().sum())
#%%
print(df_copy[df_copy['user_id']==4325]['employment_type'])
#%%
a = df_copy[df_copy['purpose'].isnull()==True]
