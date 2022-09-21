# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 19:08:46 2022

@author: yoontaejun
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
#%%
# 경로지정
os.chdir('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908')
os.getcwd()
#%%
#숫자 지수표현없이 출력



#%%
loan_result = pd.read_csv('loan_result.csv')
log_data = pd.read_csv('log_data.csv')
user_spec = pd.read_csv('user_spec.csv',encoding=('UTF-8'))
left1 = pd.read_csv('left1.csv')
#%%
print(loan_result.info())
print(loan_result.isnull().sum())
print(loan_result.describe())
#%%
# lona_result, log_data 비교
print(loan_result.head())
print(log_data.head())
#%%
a_1 = list(loan_result.application_id.unique())
b_1 = list(user_spec.application_id.unique())
#%%

#%%
a= list(user_spec.user_id.unique())
a_sort = a.sort()
b = list(log_data.user_id.unique())
b_sort = b.sort()
#%%
print(a==b)
#%%
print(a.duplicated(b))
#%%
print(log_data.info())
print(log_data.isnull().sum())
print(log_data.describe())
#%%
print(user_spec.info())
print(user_spec.isnull().sum())
print(user_spec.describe())
#%%
#론 데이터 박스플랏
f, ax = plt.subplots(figsize=(16, 14))
ax.set_xscale("log")
ax = sns.boxplot(data = loan_result , orient="h", palette="Set1")

ax.xaxis.grid(False)

plt.xlabel("Numeric values", fontsize = 10)
plt.ylabel("Feature names", fontsize = 10)
plt.title("Numeric Distribution of Features", fontsize = 15)
sns.despine(trim = True, left = True)
#%%
sort_loan = loan_result.sort_values('application_id')
sort_log = log_data.sort_values('user_id')
sort_user = user_spec.sort_values('application_id')
#%%
outer1 = pd.merge(sort_loan, sort_user, left_on='application_id', 
                 right_on='application_id', how='outer')
print(outer1.head())
#%%
sort_left = left1.sort_values('application_id')
#%%
a = sort_left.isnull().sum()
a.to_csv('레프트 조인 결측치 수.csv ')
#%%
a = sort_left.describe(include = 'all')
a.to_csv('레프트 조인 기술통계량.csv')