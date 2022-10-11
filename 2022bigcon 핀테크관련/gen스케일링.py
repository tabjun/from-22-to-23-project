# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:16:13 2022

@author: 215-01
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.stats.multicomp import MultiComparison
import scipy.stats
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm
import random
from sklearn import linear_model  # logistic regression model
from statsmodels.stats.outliers_influence import variance_inflation_factor
#%%
# 경로지정
os.chdir('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908')
os.getcwd()
#%%
# py , ipynb 변환
import subprocess
filename = r'C:\\Users\\215-01\\Desktop\\빅콘\\최종모델링.py'
dest = r'C:\\Users\\215-01\\Desktop\\빅콘\\gen스케일.ipynb'
subprocess.run(['ipynb-py-convert', filename, dest])
#%%
#숫자 지수표현없이 출력
pd.options.display.float_format = '{:.2f}'.format
#%%
'''
test_unu_loan=pd.read_csv('test_unu_loan_결측처리완.csv')
test_unu_age = pd.read_csv('test_unu_age_결측처리완.csv')
test_unu_enter=pd.read_csv('test_unu_enter_결측처리완.csv')
test_unu_credit=pd.read_csv('test_unu_credit_결측처리완.csv')

train_unu_loan=pd.read_csv('train_unu_loan_결측처리완.csv')
train_unu_age=pd.read_csv('train_unu_age_결측처리완.csv')
train_unu_enter=pd.read_csv('train_unu_enter_결측처리완.csv')
train_unu_credit=pd.read_csv('train_unu_credit_결측처리완.csv')
'''
#%%
test_gen_loan=pd.read_csv('test_gen_loan_결측처리완.csv')
test_gen_age = pd.read_csv('test_gen_age_결측처리완.xls')
test_gen_enter=pd.read_csv('test_gen_enter_결측처리완.csv')
test_gen_credit=pd.read_csv('test_gen_credit_결측처리완.csv')
#%%
train_gen_loan=pd.read_csv('train_gen_loan_결측처리완.csv')
train_gen_age=pd.read_csv('train_gen_age_결측처리완.csv')
train_gen_enter=pd.read_csv('train_gen_enter_결측처리완.csv')
train_gen_credit=pd.read_csv('train_gen_credit_결측처리완.csv')
#%%
# train_gen_age 스케일링을 위한 수치형, 범주형 나누기
train_gen_age_num = train_gen_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']]
#%%
train_gen_age_ob = train_gen_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월'],axis=1)

#%%
# test_gen_age 스케일링을 위한 수치형, 범주형 나누기
test_gen_age_num = test_gen_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']]
#%%
test_gen_age_ob = test_gen_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월'],axis=1)
#%%
# 이상치가 존재하므로 수치형 변수 gen_age 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_gen_age) # 결측치 없는 train데이터들로 fit시키고
train_gen_age_scaled = rbs.fit_transform(train_gen_age_num) #fit시킨 데이터 적용
test_gen_age_scaled = rbs.transform(test_gen_age_num) #fit시킨 데이터 적용
#%%
# 배열 형태로 반환되므로, 데이터 프레임으로 변환
train_gen_age_scaled = pd.DataFrame(data = train_gen_age_scaled )
test_gen_age_scaled = pd.DataFrame(data = test_gen_age_scaled)
#%%
# 변수명 삽입
train_gen_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']
#%%
test_gen_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']
#%%
# 나눠준 데이터 합치기 concat
train_gen_age_sca = pd.concat([train_gen_age_ob,train_gen_age_scaled],axis=1)
test_gen_age_sca = pd.concat([test_gen_age_ob,test_gen_age_scaled],axis=1)
#%%
train_gen_age_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\train_gen_age_sca.csv',index=False)
test_gen_age_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\test_gen_age_sca.csv',index=False)


#%%
# train_gen_enter 스케일링을 위한 수치형, 범주형 나누기
train_gen_enter_num = train_gen_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'gender','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']]
#%%
train_gen_enter_ob = train_gen_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'gender','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age'],axis=1)

#%%
# test_gen_enter 스케일링을 위한 수치형, 범주형 나누기
test_gen_enter_num = test_gen_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'gender','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']]
#%%
test_gen_enter_ob = test_gen_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'gender','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age'],axis=1)
#%%
# 이상치가 존재하므로 수치형 변수 gen_enter 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_gen_enter) # 결측치 없는 train데이터들로 fit시키고
train_gen_enter_scaled = rbs.fit_transform(train_gen_enter_num) #fit시킨 데이터 적용
test_gen_enter_scaled = rbs.transform(test_gen_enter_num) #fit시킨 데이터 적용
#%%
# 배열 형태로 반환되므로, 데이터 프레임으로 변환
train_gen_enter_scaled = pd.DataFrame(data = train_gen_enter_scaled )
test_gen_enter_scaled = pd.DataFrame(data = test_gen_enter_scaled)
#%%
# 변수명 삽입
train_gen_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'gender','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']
#%%
test_gen_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'gender','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']
#%%
# 나눠준 데이터 합치기 concat
train_gen_enter_sca = pd.concat([train_gen_enter_ob,train_gen_enter_scaled],axis=1)
test_gen_enter_sca = pd.concat([test_gen_enter_ob,test_gen_enter_scaled],axis=1)
#%%
train_gen_enter_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\train_gen_enter_sca.csv',index=False)
test_gen_enter_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\test_gen_enter_sca.csv',index=False)

#%%
# train_gen_loan 스케일링을 위한 수치형, 범주형 나누기
train_gen_loan_num = train_gen_loan.copy()[['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
train_gen_loan_ob = train_gen_loan.copy().drop(['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)

#%%
# test_gen_loan 스케일링을 위한 수치형, 범주형 나누기
test_gen_loan_num = test_gen_loan.copy()[['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_gen_loan_ob = test_gen_loan.copy().drop(['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
#%%
# 이상치가 존재하므로 수치형 변수 gen_loan 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_gen_loan) # 결측치 없는 train데이터들로 fit시키고
train_gen_loan_scaled = rbs.fit_transform(train_gen_loan_num) #fit시킨 데이터 적용
test_gen_loan_scaled = rbs.transform(test_gen_loan_num) #fit시킨 데이터 적용
#%%
# 배열 형태로 반환되므로, 데이터 프레임으로 변환
train_gen_loan_scaled = pd.DataFrame(data = train_gen_loan_scaled )
test_gen_loan_scaled = pd.DataFrame(data = test_gen_loan_scaled)
#%%
# 변수명 삽입
train_gen_loan_scaled.columns = ['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_gen_loan_scaled.columns = ['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
# 나눠준 데이터 합치기 concat
train_gen_loan_sca = pd.concat([train_gen_loan_ob,train_gen_loan_scaled],axis=1)
test_gen_loan_sca = pd.concat([test_gen_loan_ob,test_gen_loan_scaled],axis=1)
#%%
train_gen_loan_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\train_gen_loan_sca.csv',index=False)
test_gen_loan_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\test_gen_loan_sca.csv',index=False)



