# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 20:15:52 2022

@author: 215-01
"""

# -*- coding: utf-8 -*-


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
# 경로지정
os.chdir('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908')
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
df = pd.read_csv('고객유형.csv')
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
#85개 공통 결측 행 제거, 0.006%의 비율을 차지하고 있는 극 소량의 데이터이며, 조회과정에서
# 소득정보 입력에 대해 무관이라고 선택을 할 수도 있음.
df_copy = df.copy()
df_copy.dropna(subset=['employment_type'],inplace = True)
print(df_copy.isnull().sum())
#%%
# yearly_income을 채우는 과정
# 얘가 매번 연봉도 다르고, 대출 조건 조회 기간마다 입사년월이 달라짐, 이상한 놈
a = df_copy[df_copy['user_id']==702899]
#%%
# 예 아니오 값만 하기 위해 0이면 0 0보다 크면 1로 변경
df['수입_형태'] = df['is_applied'].apply(lambda x: 1 if x >= 1 else 0 if x == 0 else x)
#%%
# 대출 신청을 했지만, 정확한 분류에 방해를 주는 것 같으므로, id =  702899행 제거
#df_null_id = df_copy[df_copy['user_id']!=702899]
#print(df_null_id)
#%%
#print(df_null_id.isnull().sum())
#%%
#df_null_id['yearly_income'].fillna(0,inplace = True)
#%%
#df1 = df_null_id.copy()
#%%
# 입사년월과 탄생년도 비교를 위해 입사연도와 입사월으로 분리해주기
df_over = df1[df1['company_enter_month']>=202207].sort_values(['company_enter_month']) 
df_under = df1[df1['company_enter_month']<202207].sort_values(['company_enter_month'])
df_null = df1[df1['company_enter_month'].isnull()==True] 
#%%
df_under['입사_년도'] = df_under['company_enter_month']//100
df_under['입사_월'] = df_under['company_enter_month']%100
#%%
df_over['입사_년도'] = df_over['company_enter_month']//10000
df_over['입사_월'] = df_over['company_enter_month']%10000
#%%
df_1 = pd.concat([df_under, df_over], axis = 0)
df_2 = pd.concat([df_1, df_null], axis = 0)
#%%
a = df[df['personal_rehabilitation_yn']!=1].copy()
a
