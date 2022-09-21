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
left1 = pd.read_csv('left1.csv')
#%%
print(loan_result.info())
print(loan_result.isnull().sum())
print(loan_result.describe())
#%%
print(log_data.info())
print(log_data.isnull().sum())
print(log_data.describe())
#%%
print(user_spec.info())
print(user_spec.isnull().sum())
print(user_spec.describe())
#%%
sort_loan = loan.sort_values('application_id')
sort_log = log.sort_values('user_id')
sort_user = user.sort_values('application_id')
#%%
outer1 = pd.merge(sort_loan, sort_user, left_on='application_id', 
                 right_on='application_id', how='outer')
print(outer1.head())
#%%
sort_left = left1.sort_values('application_id')
sort_left.to_csv('sort_left.csv')
#%%
print(sort_left.isnull().sum())
#%%
# 결측치 제거를 위한 non_null추출
left_copy = sort_left.copy()
print((left_copy.isnull().any(axis=1)).value_counts())
print(left_copy.isnull().any(axis=0))
#%%
df_null = left_copy.dropna(axis=0)
print(df_null.info())
#%%
# 'personal_rehabilitation_yn' 살펴보기
model = ols('personal_rehabilitation_yn ~ product_id', df_null).fit()
print(anova_lm(model))

#%%
model = ols('personal_rehabilitation_yn ~ product_id + loan_limit + loan_rate + is_applied + gender + credit_score + yearly_income + houseown_type + desired_amount + purpose +  existing_loan_amt+ personal_rehabilitation_complete_yn + existing_loan_cnt', df_null).fit()
print(anova_lm(model))
#%%
logis = sm.Logit.from_formula('prodtaken ~ preferredpropertystar', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
left1['신용등급'] = left1['credit_score'].apply(lambda x: 1 if 942 <=  x <= 1000 
                                                else 2 if 891<= x <= 941 else 3 if 832<=x<=890 else 4 if 768 <= x <= 831
                                               else 5 if 698 <= x <= 767 else 6 if 630 <= x <= 697 else 7 if 530 <= x <= 629
                                               else 8 if 454 <= x <= 529 else 9 if 335 <= x <= 453 else 10 if 0 <= x <= 334 else x)
#%%
print(left1[left1['신용등급']==1]['credit_score'].sort_values())
#%%
print(left1[left1['신용등급']==2]['credit_score'])
#%%
print(left1[left1['신용등급']==3]['credit_score'])
#%%
print(left1[left1['신용등급']==4]['credit_score'])
#%%
print(left1[left1['신용등급']==5]['credit_score'])
#%%
print(left1.isnull().sum())
#%%
#결측치처리 방법 토론
#데이터 셋 자체가 용량도 크고 각 변수마다 결측치들이 존재하여 데이터 자체로 처리하기 어려움
# 도메인 지식 기반으로 처리를 해나가며 거기에 뒷받침되는 분석 결과(회귀를 통한유의확률 등)
# 또는 분석결과가 유의하지 않을시에도 논문 등으로 커버를 칠 수 있도록 해야 한다. 

# 핀다에 전화해서 데이터 분석 핀다데이터를 하는데 , 기대출 상품 이용중인데, 금액은 0 이다.
# 좀 헷갈려서 그런데 우리는 저 2개 변수가 모두 신용점수에 영향을 줄거라고 생각하는데
# 혹시 핀다 직원님은 어떻게 생각하세요?
# 건강보험이나
#%%
#일단 113개의 결측치들은 user_spec에는 없는 행이라서 합쳐지는 과정에서 생긴 결측치 그래서 그냥 제거
left_copy = left1.copy()
left = left_copy.dropna(subset = ['user_id'])
print(left.isnull().sum())
#%%
left = left.drop('Unnamed: 0',axis=1)
#%%
# yearly_income 6개 결측치가 존재하는데 그게 다 같은 사람이다 나이는 42세다.
# 신용등급은 2 , 대출목적은 사업자금, 직업은 기타, 기타가족소유
# 같은 사람이 신청한 다른 값을 보았을 때 소득이 다 0
left['yearly_income'] = left['yearly_income'].fillna(0)
#%%
#론 데이터 박스플랏
f, ax = plt.subplots(figsize=(16, 14))
ax.set_xscale("log")
ax = sns.boxplot(data = left , orient="h", palette="Set1")

ax.xaxis.grid(False)

plt.xlabel("Numeric values", fontsize = 10)
plt.ylabel("Feature names", fontsize = 10)
plt.title("Numeric Distribution of Features", fontsize = 15)
sns.despine(trim = True, left = True)

#%%
#left['입사년도'] = left['company_enter_month']//100
#left['입사월'] = left['company_enter_month']%100
#print(left['입사년도'])
#print(left['입사월'])
#%%
left = left({'입사연도'})
tr = tr.astype({'age':'int'})