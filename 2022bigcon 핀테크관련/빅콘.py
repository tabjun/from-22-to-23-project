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
from pycaret.classification import *
from sklearn.preprocessing import RobustScaler
#%%
# 경로지정
os.chdir('C:\\Users\\user\\Desktop\\경진대회\\빅콘테스트\\2022 빅콘_ 데이터분석_퓨처스리그\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908')
os.getcwd()
#%%
#숫자 지수표현없이 출력
pd.options.display.float_format = '{:.2f}'.format
#%%
loan_result = pd.read_csv('loan_result.csv')
log_data = pd.read_csv('log_data.csv')
user_spec = pd.read_csv('user_spec.csv',encoding=('UTF-8'))
#%%
print(loan_result.info())
print(loan_result.isnull().sum())
loan_describe = loan_result.describe()
loan_object_describe = loan_result.describe(include='object')
#%%
loan_describe.to_csv('C:\\Users\\user\\Desktop\\경진대회\\빅콘테스트\\2022 빅콘_ 데이터분석_퓨처스리그\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\loan_result수치형 요약.csv')
loan_object_describe.to_csv('C:\\Users\\user\\Desktop\\경진대회\\빅콘테스트\\2022 빅콘_ 데이터분석_퓨처스리그\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\loan_result문자형 요약.csv')
#%%
print(log_data.info())
print(log_data.isnull().sum())
log_describe = log_data.describe()
log_object_describe = log_data.describe(include='object')
#%%
log_describe.to_csv('C:\\Users\\user\\Desktop\\경진대회\\빅콘테스트\\2022 빅콘_ 데이터분석_퓨처스리그\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\log_data수치형 요약.csv')
log_object_describe.to_csv('C:\\Users\\user\\Desktop\\경진대회\\빅콘테스트\\2022 빅콘_ 데이터분석_퓨처스리그\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\log_data문자형 요약.csv')
#%%
print(user_spec.info())
print(user_spec.isnull().sum())
user_describe = user_spec.describe()
user_object_describe = user_spec.describe(include='object')
#%%
user_describe.to_csv('C:\\Users\\user\\Desktop\\경진대회\\빅콘테스트\\2022 빅콘_ 데이터분석_퓨처스리그\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\user_spec수치형 요약.csv')
user_object_describe.to_csv('C:\\Users\\user\\Desktop\\경진대회\\빅콘테스트\\2022 빅콘_ 데이터분석_퓨처스리그\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\user_spec문자형 요약.csv')
#%%
#datetime변환을 이용하여 6월 데이터 개수 확인
loan_result['loanapply_insert_time'] = pd.to_datetime(loan_result['loanapply_insert_time'])
#%%
loan_result['month'] = pd.DatetimeIndex(loan_result['loanapply_insert_time']).month
#%%
print(loan_result[loan_result['month']==6])
#%%
loan_result.to_csv('C:\\Users\\user\\Desktop\\경진대회\\빅콘테스트\\2022 빅콘_ 데이터분석_퓨처스리그\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\loan_result_월별분리.csv')
