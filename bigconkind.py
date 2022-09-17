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
import p2j
#%%
# 경로지정
os.chdir('C:\\Users\\user\\Desktop\\경진대회\\빅콘테스트\\2022 빅콘_ 데이터분석_퓨처스리그\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908')
os.getcwd()
#%%
#숫자 지수표현없이 출력
pd.options.display.float_format = '{:.2f}'.format
#%%
# 플랏 한글
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#%%
loan = pd.read_csv('loan_result.csv')
log = pd.read_csv('log_data.csv')
user = pd.read_csv('user_spec.csv',encoding=('UTF-8'))
#%%
# user is_applied변수를 만드는데 loan_application_id를 기준으로 is_applied
user['is_applied'] = loan.groupby(['application_id'])['is_applied'].transform('sum')
#%%
print(user[user['is_applied']>=1])
print(user.info())
#%%
# 예 아니오 값만 하기 위해 0이면 0 0보다 크면 1로 변경
user['대출신청여부'] = user['is_applied'].apply(lambda x: 1 if x >= 1 else 0 if x == 0 else x)
#%%
user = user.drop('대출신청여',axis=1)
print(user.info())
#%%
print(user['대출신청여부'].value_counts())
#%%
ipynb-py-convert C:/Users/user/Desktop/경진대회/빅콘테스트/2022 빅콘_ 데이터분석_퓨처스리그/bigconkind.py C:/Users/user/Desktop/경진대회/빅콘테스트/2022 빅콘_ 데이터분석_퓨처스리그/bigcon.ipynb
#%%
#ipynb로 바꾸기
import subprocess

filename = r'C:/Users/user/Desktop/경진대회/빅콘테스트/2022 빅콘_ 데이터분석_퓨처스리그/bigconkind.py'
dest = r'C:/Users/user/Desktop/경진대회/빅콘테스트/2022 빅콘_ 데이터분석_퓨처스리그/bigconkind.ipynb'
subprocess.run(['ipynb-py-convert', filename, dest])