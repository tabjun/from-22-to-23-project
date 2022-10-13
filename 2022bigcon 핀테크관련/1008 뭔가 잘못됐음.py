# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:09:17 2022

@author: 215-01
"""

'''
고객유형으로 분류하고, 예측을 내놓게 된다면, 상품 종류에 관계없이, 0과 1로 예측함.

유형 분류보다 상품분류를 기준으로 분석을 하고, 지금까지 하던 고객유형별 데이터셋을
들고 분석하는게 좋을 듯
이 때 6월이냐 아니냐에 & 개인회생 여부 따라 train,test나누기
'''

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
#숫자 지수표현없이 출력
pd.options.display.float_format = '{:.2f}'.format
#%%
# py , ipynb 변환
import subprocess
filename = r'C:\\Users\\215-01\\Desktop\\빅콘\\1008 뭔가 잘못됐음.py'
dest = r'C:\\Users\\215-01\\Desktop\\빅콘\\bigcon_1008.ipynb'
subprocess.run(['ipynb-py-convert', filename, dest])
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
print(loan.info()) # 데이터가 너무 커서 dtype이 안나옴
print(loan.dtypes)
print(loan.isnull().sum())
print(loan.describe(include='all'))
#%%
print(log.info())
print(log.dtypes)
print(log.isnull().sum())
print(log.describe(include='all'))
#%%
print(user.info())
print(user.dtypes)
print(user.isnull().sum())
print(user.describe(include='all'))
#%%
#log_data의 어플 버전, 안드로이드 ios구분, 값이 다르다. 
# 태경 안드로이드 폰으로 버전 보여주고, 소문자 값이 휴대폰인지 확인
print(log['mp_os'].value_counts())
print(log['mp_app_version'].value_counts())

#%%
# left join으로 진행
# user 와 loan 데이터 셋의 application_id 정렬 이후 
sort_loan = loan.sort_values('application_id')
sort_log = log.sort_values('user_id')
sort_user = user.sort_values('application_id')

left = pd.merge(sort_loan, sort_user, left_on='application_id', 
                 right_on='application_id', how='left')
print(left.isnull().sum())
#%%
left_copy = left.copy()
#%%
#datetime변환을 이용하여 6월 데이터 개수 확인
left_copy['loanapply_insert_time'] = pd.to_datetime(left_copy['loanapply_insert_time'])
# datetime변수에서 month추출
left_copy['month'] = pd.DatetimeIndex(left_copy['loanapply_insert_time']).month
#%%
# purpose 값 바꿔주기
left_copy = left_copy.replace('LIVING','생활비')
left_copy = left_copy.replace('SWITCHLOAN', '대환대출')
left_copy = left_copy.replace('BUSINESS', '사업자금')
left_copy = left_copy.replace('ETC', '기타')
left_copy = left_copy.replace('HOUSEDEPOSIT', '전월세보증금')
left_copy = left_copy.replace('BUYHOUSE', '주택구입')
left_copy = left_copy.replace('INVEST', '투자')
left_copy = left_copy.replace('BUYCAR', '자동차구입')
#%%
left_copy.to_csv('left.csv')
#%%
left = pd.read_csv('left.csv')
left_copy = left.copy()
#%%
# 단변량 로짓으로 유의성 판단, 이 때 개인회생 여부로 나눠서, 일반적인 경우, 
# 개인 회생만 고려할 것이 아니라, 개인 회생 납입 완료 여부도 고려해서 함께 나눠줘야 함.
# 개인 회생이 결측이라도, 개인 회생 납입 여부 == 1, 이면 개인 회생 신청한 애들
# 개인 회생을 신청했어도, 변제 금액을 다 갚고, 신용 점수 회복한 애들(5~6등급)
# 다시 일반적인 범위로 들어온 애들
# 아닌 경우도 함께 파악
# 개인회생 !=1 & !6월 = general train셋, 개인회생 ==1 & !6월 = unusual train셋
# 개인회생 !=1 & 6월 = 일반적인 test셋, 개인회생 ==1 & 6월 = 일반적이지 않은 test셋

# unusual train set, 6월이 아니면서, 개인회생이 1이거나, 개인 회생 납입 완료 여부 1
train_unu = left_copy[(left_copy['month']!=6)&
                      ((left_copy['personal_rehabilitation_yn']==1)|
                       (left_copy['personal_rehabilitation_complete_yn']==1))]
# general train set, 6월이 아니면서, 개인회생이 !=1 이면서, 개인 회생 납입 완료 여부 != 1
train_gen = left_copy[(left_copy['month']!=6)&
                      ((left_copy['personal_rehabilitation_yn']!=1)&
                       (left_copy['personal_rehabilitation_complete_yn']!=1))]
# unusual train set, 6월이면서, 개인회생이 1이거나, 개인 회생 납입 완료 여부 1
test_unu = left_copy[(left_copy['month']==6)&
                     ((left_copy['personal_rehabilitation_yn']==1)|
                       (left_copy['personal_rehabilitation_complete_yn']==1))]
# unusual train set, 6월이 아니면서, 개인회생이 1이거나, 개인 회생 납입 완료 여부 1
test_gen = left_copy[(left_copy['month']==6)&
                     ((left_copy['personal_rehabilitation_yn']!=1)&
                     (left_copy['personal_rehabilitation_complete_yn']!=1))]

#%%
# gen의 개인 회생, 납입 완료 값 확인 0 또는 결측만 있음, 잘 담김
print('일반 훈련셋')
print(train_gen[['personal_rehabilitation_yn','personal_rehabilitation_complete_yn']].value_counts())
print('일반 테스트 셋')
print(test_gen[['personal_rehabilitation_yn','personal_rehabilitation_complete_yn']].value_counts())
print('특별 훈련셋')
print(train_unu[['personal_rehabilitation_yn','personal_rehabilitation_complete_yn']].value_counts())
print('특별 테스트')
print(test_unu[['personal_rehabilitation_yn','personal_rehabilitation_complete_yn']].value_counts())
#%%
print(train_unu.personal_rehabilitation_complete_yn.value_counts())
print(test_unu.personal_rehabilitation_complete_yn.value_counts())
# 결측치 종류 MAR, 개인 회생 납입 완료 여부 변수, 1을 제외하곤 개인회생조차 신청안했거나,
# 개인회생을 신청안했지만 납입 하지 않은 경우, 결측치 0으로 채워주면 됨
print('결측치 개수 파악')
train_unu['personal_rehabilitation_complete_yn'].fillna(0,inplace = True)
test_unu['personal_rehabilitation_complete_yn'].fillna(0,inplace = True)
print(f'개인회생납입 완료 여부 결측 개수: {train_unu.personal_rehabilitation_complete_yn.isnull().sum()}')
print(f'test개인회생납입 완료 여부 결측 개수: {test_unu.personal_rehabilitation_complete_yn.isnull().sum()}')
#%%
#train셋의 user id로 탄생년도와 성별 채우기
# 두 변수는 개인의 고유적인 특징, 불변이기 때문에 채워줌
train_gen['birth_year'] = train_gen['birth_year'].fillna(train_gen.groupby('user_id')['birth_year'].transform('mean'))
train_gen['gender'] = train_gen['gender'].fillna(train_gen.groupby('user_id')['gender'].transform('mean'))
#%%
test_gen['birth_year'] = test_gen['birth_year'].fillna(train_gen.groupby('user_id')['birth_year'].transform('mean'))
test_gen['gender'] = test_gen['gender'].fillna(train_gen.groupby('user_id')['gender'].transform('mean'))
#%%
train_unu['birth_year'] = train_unu['birth_year'].fillna(train_unu.groupby('user_id')['birth_year'].transform('mean'))
train_unu['gender'] = train_unu['gender'].fillna(train_unu.groupby('user_id')['gender'].transform('mean'))
#%%
test_unu['birth_year'] = train_unu['birth_year'].fillna(train_unu.groupby('user_id')['birth_year'].transform('mean'))
test_unu['gender'] = train_unu['gender'].fillna(train_unu.groupby('user_id')['gender'].transform('mean'))
#%%
# 나눠진 데이터 셋들의 정보에는, 개인 회생을 신청한 사람과, 신청하지 않은 사람의 정보가 담겨있음.
# 정보가 다 담겨 있기 때문에, 굳이 결측을 채워주지 않고, 변수를 제거하고 사용
# 개인회생 여부 포함하면서 나눠줌, gen, unu 셋에는 개인회생을 신청한 애들과, 신청안한 애들
# 나눠짐, 변수 제거해도 그 속성은 남아있어서 변수 제거
# unu는 개인회생 완납, 미납 차이있는지 살펴봐야해서 완납은 살려 둠
train_unu.drop(['personal_rehabilitation_yn'],axis=1,inplace =True)
train_gen.drop(['personal_rehabilitation_yn','personal_rehabilitation_complete_yn'],axis=1,inplace =True)
test_unu.drop(['personal_rehabilitation_yn'],axis=1,inplace =True)
test_gen.drop(['personal_rehabilitation_yn','personal_rehabilitation_complete_yn'],axis=1,inplace =True)
#%%

# 단변량 로지스틱
# n이 많아서 유의확률 낮음
'''
logis_gen_birth = sm.Logit.from_formula('is_applied ~ birth_year', train_gen).fit()
print(logis_gen_birth.summary())
print(np.exp(logis_gen_birth.params)) #  로짓값 출력
print(f'train_gen:{logis_gen_birth.aic}')

logis_unu_birth = sm.Logit.from_formula('is_applied ~ birth_year', train_unu).fit()
print(logis_unu_birth.summary())
print(np.exp(logis_unu_birth.params)) #  로짓값 출력
print(f'train_unu:{logis_unu_birth.aic}')
'''
for i in train_gen.columns:
    model = sm.Logit.from_formula('is_applied ~ train_gen[i]', train_gen).fit()
    print(f'독립변수 이름: {i}')
    print(logis_gen_birth.summary())
    print('============='*3)
    print(f'train_gen:{logis_gen_birth.aic}')
    print('\n')
    print(np.exp(logis_gen_birth.params))
#%%
'''
기대출수가 존재하지만, 기대출금액이 0인 경우: 금융사에서 금액을 제공하지 않은 경우
결측치 채우기 위해 확인 과정
''' 
print(train_gen[(train_gen['existing_loan_amt']==0)&(train_gen['existing_loan_cnt']>=1)])
print('\t')
print(test_gen[(test_gen['existing_loan_amt']==0)&(test_gen['existing_loan_cnt']>=1)])
print('\t')
print(train_unu[(train_unu['existing_loan_amt']==0)&(train_unu['existing_loan_cnt']>=1)])
print('\t')
print(test_unu[(test_unu['existing_loan_amt']==0)&(test_unu['existing_loan_cnt']>=1)])

#%%
print(train_gen[(train_gen['existing_loan_amt'].isnull())&(train_gen['existing_loan_cnt'].isnull())])
print('\t')
print(test_gen[(test_gen['existing_loan_amt'].isnull())&(test_gen['existing_loan_cnt'].isnull())])
print('\t')
print(train_unu[(train_unu['existing_loan_amt'].isnull())&(train_unu['existing_loan_cnt'].isnull())])
print('\t')
print(test_unu[(test_unu['existing_loan_amt'].isnull())&(test_unu['existing_loan_cnt'].isnull())])
#%%
# cnt 변수의 nan 값은 기대출 수가 없는 사람으로 판단하고 cat, amt 변수들의 nan값들을 다 0으로 채워줌
train_gen.loc[train_gen['existing_loan_cnt'] != train_gen['existing_loan_cnt'], 'existing_loan_cnt'] = 0
train_gen.loc[train_gen['existing_loan_amt'] != train_gen['existing_loan_amt'], 'existing_loan_amt'] = 0
# 기대출 수가 있지만 금융사에서 제공해주지 않아 amt변수의 값이 0인 것들은 기대출수의 따른 평균 값으로 대체
train_gen.loc[(train_gen['existing_loan_cnt'] == 1) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 35353221
train_gen.loc[(train_gen['existing_loan_cnt'] == 2) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 57461998
train_gen.loc[(train_gen['existing_loan_cnt'] == 3) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 75134303
train_gen.loc[(train_gen['existing_loan_cnt'] == 4) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 90688457
train_gen.loc[(train_gen['existing_loan_cnt'] == 5) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 105104832
train_gen.loc[(train_gen['existing_loan_cnt'] == 6) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 113855294
train_gen.loc[(train_gen['existing_loan_cnt'] == 13) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 151771285
#%%
# 조건에 맞는 행 제거
# 2022년11월인 애, is_applied가 0, 데이터 셋 자체가 크기 때문에, 수치 강제 변동 보다는
# 48개 있으므로 제거가 좋을듯, is_applied에 0 이 대부분이라서 굳이 제거해도 크게 지장없을 듯
idx = train_gen[train_gen['company_enter_month'] == 202211].index
train_gen.drop(idx , inplace=True) 
#%%
# user_id 113개 행 결측 제거
train_gen.dropna(subset=['user_id'],axis = 0,inplace = True)
#%%
# company_enter_month 6자리, 8자리 어지러움, 근속년수, 근속일 등으로 다르게 이용
# 입사년월이 6자리인 애들
train_gen['입사_년도'] = train_gen['company_enter_month']//100
train_gen['입사_월'] = train_gen['company_enter_month']%100
#%%
# 확인했을 때 잘 분리됨
print(train_gen['입사_년도'].describe()) 
print(train_gen['입사_월'].describe())
#%%
# train_unu 년월 분리
train_unu['입사_년도'] = train_unu['company_enter_month']//100
train_unu['입사_월'] = train_unu['company_enter_month']%100
#%%
# 확인했을 때 잘 분리됨
print(train_unu['입사_년도'].describe()) 
print(train_unu['입사_월'].describe())
#%%
# test_gen 년월 분리
test_gen_over = test_gen[test_gen['company_enter_month']>=202207]
test_gen_under = test_gen[test_gen['company_enter_month']<202207]
test_gen_null = test_gen[test_gen['company_enter_month'].isnull()==True]
#%%
# 입사년월이 6자리인 애들
test_gen_under['입사_년도'] = test_gen_under['company_enter_month']//100
test_gen_under['입사_월'] = test_gen_under['company_enter_month']%100
#%%
# 확인했을 때 잘 분리됨
print(test_gen_under['입사_년도'].describe()) 
print(test_gen_under['입사_월'].describe())
#%%
# 입사년월이 8자리인 애들
test_gen_over['입사_년도'] = test_gen_over['company_enter_month']//10000
test_gen_over['입사_월'] = (test_gen_over['company_enter_month']//100)%100
#%%
# 확인했을 때 잘 분리됨
print(test_gen_over['입사_년도'].describe()) 
print(test_gen_over['입사_월'].describe())
#%%
# 각각 입사년월 나눠준 데이터 셋 합치기
test_gen_1 = pd.concat([test_gen_under, test_gen_over], axis = 0)
test_gen = pd.concat([test_gen_1, test_gen_null], axis = 0)
#%%
# 확인, 잘 됨
print(test_gen['입사_년도'].describe())
print(test_gen['입사_월'].describe())
#%%
# test_unu 분리
test_unu_over = test_unu[test_unu['company_enter_month']>=202207]
test_unu_under = test_unu[test_unu['company_enter_month']<202207]
test_unu_null = test_unu[test_unu['company_enter_month'].isnull()==True]
#%%
# 입사년월이 6자리인 애들
test_unu_under['입사_년도'] = test_unu_under['company_enter_month']//100
test_unu_under['입사_월'] = test_unu_under['company_enter_month']%100
#%%
# 확인했을 때 잘 분리됨
print(test_unu_under['입사_년도'].describe()) 
print(test_unu_under['입사_월'].describe())
#%%
# 입사년월이 8자리인 애들
test_unu_over['입사_년도'] = test_unu_over['company_enter_month']//10000
test_unu_over['입사_월'] = (test_unu_over['company_enter_month']//100)%100
#%%
# 확인했을 때 잘 분리됨
print(test_unu_over['입사_년도'].describe()) 
print(test_unu_over['입사_월'].describe())
#%%
# 각각 입사년월 나눠준 데이터 셋 합치기
test_unu_1 = pd.concat([test_unu_under, test_unu_over], axis = 0)
test_unu = pd.concat([test_unu_1, test_unu_null], axis = 0)
#%%
# 확인, 잘 됨
print(test_unu['입사_년도'].describe())
print(test_unu['입사_월'].describe())
#%%
# 근속으로 만들기
train_gen['근속년도'] = 2022 - train_gen['입사_년도']  
train_gen['근속개월'] = train_gen['근속년도']*12 + train_gen['입사_월']  
#%%
train_unu['근속년도'] = 2022 - train_unu['입사_년도']  
train_unu['근속개월'] = train_unu['근속년도']*12 + train_unu['입사_월']
#%%
test_gen['근속년도'] = 2022 - test_gen['입사_년도']  
test_gen['근속개월'] = test_gen['근속년도']*12 + test_gen['입사_월']
#%%
test_unu['근속년도'] = 2022 - test_unu['입사_년도']  
test_unu['근속개월'] = test_unu['근속년도']*12 + test_unu['입사_월']
#%%
train_gen['age'] = 2022 - train_gen['birth_year']  
train_unu['age'] = 2022 - train_unu['birth_year']  
test_gen['age'] = 2022 - test_gen['birth_year']  
test_unu['age'] = 2022 - test_unu['birth_year']  
#%%
train_gen.to_csv('train_gen_1013.csv',encoding = 'cp949')
train_unu.to_csv('train_unu_1013.csv',encoding = 'cp949')
test_gen.to_csv('test_gen_1013.csv',encoding = 'cp949')
test_unu.to_csv('test_unu_1013.csv',encoding = 'cp949')
#%%
print(train_gen.isnull().sum())
print('\n')
print(train_unu.isnull().sum())
print('\n')
print(test_gen.isnull().sum())
print('\n')
print(test_unu.isnull().sum())
#%%
print(train_gen.dtypes)
#%%
train_gen = pd.read_csv('train_gen.csv',encoding='cp949')
train_unu = pd.read_csv('train_unu.csv',encoding='cp949')
test_gen = pd.read_csv('test_gen.csv',encoding='cp949')
test_unu = pd.read_csv('test_unu.csv',encoding='cp949')
#%%
# 모델링 과정, 변수 버리기, 수치형들만 놔두기
# 라벨 인코딩을 해야하므로 결국 범주형 변수들도 수치형으로 사용해야 함
# 변수 내 1,2,3,,,,200까지 있기 때문에 스케일링을 통해 그 영향을 줄임
# type 변수는 문자형 그대로, 더미변수로 생성
# 변수변환에 사용해준 변수도 버림
train_gen.drop(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)
#%%
train_unu.drop([ 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)
#%%
test_gen.drop([ 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)
#%%
test_unu.drop([ 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)

#%%
# 데이터 나눠주기
# 1단계 train_gen에서 loan 결측 행 분리
train_gen_copy = train_gen.copy()
train_gen_loan = train_gen_copy[train_gen_copy['loan_limit'].isnull()==True] # loan만 결측치인 행 추출
train_gen_drop_loan = train_gen_copy.dropna(subset=['loan_limit'],axis = 0) # loan결측치 제거된 데이터 셋
#%%
# 2단계 train_gen에서 birth,gender 결측 행 분리
train_gen_age = train_gen_drop_loan[train_gen_drop_loan['age'].isnull()==True] # # loan결측치 제거된 데이터 셋에서 birth_year행 결측치 추출
train_gen_drop_age = train_gen_drop_loan.dropna(subset=['age'],axis = 0) # loan + birth_year 결측치가 존재하지 않는 데이터 셋
#%%
# 3단계 train_gen에서 enter_month 결측 행 분리
train_gen_enter = train_gen_drop_age[train_gen_drop_age['근속개월'].isnull()==True] # company_enter_month만 결측치인 행 추출
train_gen_drop_enter = train_gen_drop_age.dropna(subset=['근속개월'],axis = 0) 
# loan + birth_year + company_enter_month결측치가 존재하지 않는 데이터 셋
#%%
# 4단계 train_gen에서 credit_score결측 행 분리
train_gen_credit = train_gen_drop_enter[train_gen_drop_enter['credit_score'].isnull()==True] # credit_score만 결측치인 행 추출
train_gen_drop_na = train_gen_drop_enter.dropna(subset=['credit_score'],axis = 0) 
# loan + birth_year + company_enter_month + credit_score 결측치가 존재하지 않는 데이터 셋
#%%
train_gen_age.to_csv('train_gen_age.csv',index = False)
train_gen_credit.to_csv('train_gen_credit.csv',index = False)
train_gen_drop_na.to_csv('train_gen_drop_na.csv',index = False)
train_gen_enter.to_csv('train_gen_enter.csv',index = False)
train_gen_loan.to_csv('train_gen_loan.csv',index = False)
#%%
train_gen_age.to_csv('train_gen_age_cp.csv',encoding = 'cp949',index = False)
train_gen_credit.to_csv('train_gen_credit_cp.csv',encoding = 'cp949',index = False)
train_gen_drop_na.to_csv('train_gen_drop_na_cp.csv',encoding = 'cp949',index = False)
train_gen_enter.to_csv('train_gen_enter_cp.csv',encoding = 'cp949',index = False)
train_gen_loan.to_csv('train_gen_loan_cp.csv',encoding = 'cp949',index = False)


#%%
# 데이터 나눠주기
# 1단계 test_gen에서 loan 결측 행 분리
test_gen_copy = test_gen.copy()
test_gen_loan = test_gen_copy[test_gen_copy['loan_limit'].isnull()==True] # loan만 결측치인 행 추출
test_gen_drop_loan = test_gen_copy.dropna(subset=['loan_limit'],axis = 0) # loan결측치 제거된 데이터 셋
#%%
# 2단계 test_gen에서 birth,gender 결측 행 분리
test_gen_age = test_gen_drop_loan[test_gen_drop_loan['age'].isnull()==True] # # loan결측치 제거된 데이터 셋에서 birth_year행 결측치 추출
test_gen_drop_age = test_gen_drop_loan.dropna(subset=['age'],axis = 0) # loan + birth_year 결측치가 존재하지 않는 데이터 셋
#%%
# 3단계 test_gen에서 enter_month 결측 행 분리
test_gen_enter = test_gen_drop_age[test_gen_drop_age['근속개월'].isnull()==True] # company_enter_month만 결측치인 행 추출
test_gen_drop_enter = test_gen_drop_age.dropna(subset=['근속개월'],axis = 0) 
# loan + birth_year + company_enter_month결측치가 존재하지 않는 데이터 셋
#%%
# 4단계 test_gen에서 credit_score결측 행 분리
test_gen_credit = test_gen_drop_enter[test_gen_drop_enter['credit_score'].isnull()==True] # credit_score만 결측치인 행 추출
test_gen_drop_na = test_gen_drop_enter.dropna(subset=['credit_score'],axis = 0) 
# loan + birth_year + company_enter_month + credit_score 결측치가 존재하지 않는 데이터 셋
#%%
test_gen_age.to_csv('test_gen_age.csv',index = False)
test_gen_credit.to_csv('test_gen_credit.csv',index = False)
test_gen_drop_na.to_csv('test_gen_drop_na.csv',index = False)
test_gen_enter.to_csv('test_gen_enter.csv',index = False)
test_gen_loan.to_csv('test_gen_loan.csv',index = False)
#%%
test_gen_age.to_csv('test_gen_age_cp.csv',encoding = 'cp949',index = False)
test_gen_credit.to_csv('test_gen_credit_cp.csv',encoding = 'cp949',index = False)
test_gen_drop_na.to_csv('test_gen_drop_na_cp.csv',encoding = 'cp949',index = False)
test_gen_enter.to_csv('test_gen_enter_cp.csv',encoding = 'cp949',index = False)
test_gen_loan.to_csv('test_gen_loan_cp.csv',encoding = 'cp949',index = False)
#%%






#%%
print(train_gen.columns)
#%%
# train_gen 스케일링을 위한 수치형, 범주형 나누기
train_gen_num = train_gen_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
train_gen_ob = train_gen_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)

#%%
# train_unu 스케일링을 위한 수치형, 범주형 나누기
test_gen_num = train_unu.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
train_unu_ob = train_unu.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)

#%%
# 이상치가 존재하므로 수치형 변수 gen 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_gen) # 결측치 없는 train데이터들로 fit시키고
train_gen_scaled = rbs.fit_transform(train_gen_num) #fit시킨 데이터 적용
test_gen_scaled = rbs.transform(test_gen_num) #fit시킨 데이터 적용
#%%
# unu 스케일링
from sklearn.preprocessing import RobustScaler
rbs.fit_transform(no_train_unu)
train_unu_scaled = rbs.transform(train_unu_num)
test_unu_scaled = rbs.transform(test_unu_num)
#%%
# 배열 형태로 반환되므로, 데이터 프레임으로 변환
train_gen_scaled = pd.DataFrame(data = train_gen_scaled )
test_gen_scaled = pd.DataFrame(data = test_gen_scaled)
#%%
train_unu_scaled = pd.DataFrame(data = train_unu_scaled)
test_unu_scaled = pd.DataFrame(data = test_unu_scaled)

#%%
# 변수명 삽입
train_gen_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
train_unu_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_gen_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_unu_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
# train_gen_ob셋은 원래 데이터 셋에서 행들을 제거해준것이기 때문에 인덱스가 일정하지 않음
# train_gen_scaled는 새로 추출해서 한 값이기에 인덱스가 1~800000까지 일정
train_gen_scaled.reset_index(drop = False, inplace = True)
train_gen_ob.reset_index(drop = False, inplace = True)

test_gen_scaled.reset_index(drop = False, inplace = True)
test_gen_ob.reset_index(drop = False, inplace = True)

#%%
# 나눠준 데이터 합치기 concat
train_gen_sca = pd.concat([train_gen_ob,train_gen_scaled],axis=1)
#%%
train_gen_sca.to_csv('train_gen_drop_na_sca_cp.csv',index=False,encoding='cp949')
train_gen_sca.to_csv('train_gen_drop_na_sca.csv',index=False)
#%%
train_unu_sca = pd.concat([train_unu_ob,train_unu_scaled],axis=1)
test_gen_sca = pd.concat([test_gen_ob,test_gen_scaled],axis=1)
test_unu_sca = pd.concat([test_unu_ob,test_unu_scaled],axis=1)
#%%
test_gen_loan.to_csv('test_gen_loan.csv')
test_gen_age.to_csv('test_gen_age.csv')
test_gen_enter.to_csv('test_gen_enter.csv')
test_gen_credit.to_csv('test_gen_credit.csv')
test_gen_drop_na.to_csv('test_gen_drop_na.csv')
#%%
test_gen_loan.to_csv('test_gen_loan_cp.csv',encoding = 'cp949')
test_gen_age.to_csv('test_gen_age_cp.csv',encoding = 'cp949')
test_gen_enter.to_csv('test_gen_enter_cp.csv',encoding = 'cp949')
test_gen_credit.to_csv('test_gen_credit_cp.csv',encoding = 'cp949')
test_gen_drop_na.to_csv('test_gen_drop_na_cp.csv',encoding = 'cp949')



#%%
#수치형 변수만 남기기
#tr_gen_num = train_gen.copy().drop(['application_id', 'loanapply_insert_time', 'bank_id', 'product_id',
#       'user_id', 'birth_year','gender', 'insert_time','income_type',
#       'company_enter_month', 'employment_type', 'houseown_type',
#       'purpose', 'existing_loan_cnt','existing_loan_amt','month','Unnamed: 0'],axis=1)

tr_gen_num = train_gen_sca.copy()[['loan_limit','loan_rate','age','credit_score','yearly_income',
                                   'desired_amount','근속개월',
                                   'existing_loan_cnt','existing_loan_amt']]

#다중공선성
tr_gen_num = tr_gen_num.dropna(axis=0)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(tr_gen_num.values, i) for i in range(tr_gen_num.shape[1])]
vif["features"] = tr_gen_num.columns
print(vif)
# 다중공선성 없음

#%%
vif.to_csv('수치형 다중 공선성.csv')

#%%
'''
gen_data 결측치 처리
'''

# In[47]:


for i in x_train.columns:
    lm = sm.OLS(y_train, x_train[i])
    results = lm.fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(results.summary())
    print('\n')

# In[8]:


train_gen_age.drop(['Unnamed: 0.1.1', 'gender', 'age'], axis = 1, inplace = True)


# In[9]:


train_gen_credit.drop(['Unnamed: 0.1.1', 'credit_score'], axis = 1, inplace = True)


# In[75]:


train_gen_enter.drop(['Unnamed: 0.1.1', '근속개월'], axis = 1, inplace = True)


# In[76]:


train_gen_loan.drop(['Unnamed: 0.1.1', 'loan_limit', 'loan_rate'], axis = 1, inplace = True)


# In[77]:


test_gen_age.drop(['Unnamed: 0.1.1', 'gender', 'age'], axis = 1, inplace = True)


# In[78]:


test_gen_credit.drop(['Unnamed: 0.1.1', 'credit_score'], axis = 1, inplace = True)


# In[79]:


test_gen_enter.drop(['Unnamed: 0.1.1', '근속개월'], axis = 1, inplace = True)


# In[80]:


test_gen_loan.drop(['Unnamed: 0.1.1', 'loan_limit', 'loan_rate'], axis = 1, inplace = True)


# In[47]:


train_gen_loan.isnull().sum()


# In[228]:


train_gen_age.isnull().sum() # age, gender 버리기. 근속개월, credit 채워야함.


# In[34]:


train_gen_credit.isnull().sum()


# In[37]:


train_gen_enter.isnull().sum()





# In[81]:


train_gen_age['credit_score'] = train_gen_age['credit_score'].fillna(train_gen_drop_na.groupby(['bank_id', 'loan_rate', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월'])['credit_score'].transform('mean'))


# In[82]:


test_gen_age['credit_score'] = test_gen_age['credit_score'].fillna(train_gen_drop_na.groupby(['bank_id', 'loan_rate', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월'])['credit_score'].transform('mean'))


# In[83]:


train_gen_age['근속개월'] = train_gen_age['근속개월'].fillna(train_gen_drop_na.groupby(['loan_rate', 'employment_type', 'houseown_type', 'desired_amount',  'purpose', 'existing_loan_cnt'])['근속개월'].transform('mean'))


# In[84]:


test_gen_age['근속개월'] = test_gen_age['근속개월'].fillna(train_gen_drop_na.groupby(['loan_rate', 'employment_type', 'houseown_type', 'desired_amount',  'purpose', 'existing_loan_cnt'])['근속개월'].transform('mean'))


# In[85]:


train_gen_age.isnull().sum()


# In[86]:


test_gen_age.isnull().sum()


# # gen_loan 채움

# In[87]:


train_gen_loan['gender'] = train_gen_loan['gender'].fillna(train_gen_drop_na.groupby(['desired_amount'])['gender'].transform('mean'))


# In[88]:


train_gen_loan['credit_score'] = train_gen_loan['credit_score'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월', 'age'])['credit_score'].transform('mean'))


# In[89]:


train_gen_loan['근속개월'] = train_gen_loan['근속개월'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['근속개월'].transform('mean'))


# In[90]:


train_gen_loan['age'] = train_gen_loan['age'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt'])['age'].transform('mean'))


# In[91]:


test_gen_loan['gender'] = test_gen_loan['gender'].fillna(train_gen_drop_na.groupby(['desired_amount'])['gender'].transform('mean'))


# In[92]:


test_gen_loan['credit_score'] = test_gen_loan['credit_score'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월', 'age'])['credit_score'].transform('mean'))


# In[93]:


test_gen_loan['근속개월'] = test_gen_loan['근속개월'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['근속개월'].transform('mean'))


# In[94]:


test_gen_loan['age'] = test_gen_loan['age'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt'])['age'].transform('mean'))


# In[95]:


train_gen_loan.isnull().sum()


# In[96]:


test_gen_loan.isnull().sum()


# # enter 채움

# In[97]:


train_gen_enter['credit_score'] = train_gen_enter['credit_score'].fillna(train_gen_drop_na.groupby(['bank_id', 'product_id', 'loan_rate', 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['credit_score'].transform('mean'))


# In[98]:

test_gen_enter['credit_score'] = test_gen_enter['credit_score'].fillna(train_gen_drop_na.groupby(['bank_id', 'product_id', 'loan_rate', 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['credit_score'].transform('mean'))


# In[99]:


train_gen_enter.isnull().sum()


# In[100]:


test_gen_enter.isnull().sum()


# # -------------------------------------------------------------------------------------------

# In[50]:


train_gen_loan['gender'] = train_gen_loan['gender'].fillna(train_gen_drop_na.groupby(['desired_amount'])['gender'].transform('mean'))


# In[53]:


train_gen_loan['credit_score'] = train_gen_loan['credit_score'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월', 'age'])['credit_score'].transform('mean'))


# In[56]:


train_gen_loan['근속개월'] = train_gen_loan['근속개월'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['근속개월'].transform('mean'))


# In[59]:


train_gen_loan['age'] = train_gen_loan['age'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt'])['age'].transform('mean'))


# In[58]:


for i in train_gen_loan.columns:
    model = ols('age ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt'


# In[55]:


for i in train_gen_loan.columns:
    model = ols('근속개월 ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'


# In[52]:


for i in train_gen_loan.columns:
    model = ols('credit_score ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월', 'age'


# In[49]:


for i in train_gen_loan.columns:
    model = ols('gender ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'desired_amount'


# In[27]:


for i in train_gen_age.columns:
    model = ols('근속개월 ~ train_gen_age[i]', train_gen_age).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'loan_rate', 'employment_type', 'houseown_type', 'desired_amount',  'purpose', 'existing_loan_cnt'


# In[28]:


train_gen_age['근속개월'] = train_gen_age['근속개월'].fillna(train_gen_drop_na.groupby(['loan_rate', 'employment_type', 'houseown_type', 'desired_amount',  'purpose', 'existing_loan_cnt'])['credit_score'].transform('mean'))


# In[29]:


train_gen_age.isnull().sum()


# In[30]:


train_gen_credit.isnull().sum()


# In[44]:


train_gen_enter.isnull().sum()


# In[43]:


train_gen_enter['credit_score'] = train_gen_enter['credit_score'].fillna(train_gen_drop_na.groupby(['bank_id', 'product_id', 'loan_rate', 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['credit_score'].transform('mean'))


# In[38]:


for i in train_gen_enter.columns:
    model = ols('credit_score ~ train_gen_enter[i]', train_gen_enter).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'bank_id', 'product_id', 'loan_rate', 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'



#%%
train_gen_age = pd.read_csv('train_gen_age결측처리.csv')
test_gen_age = pd.read_csv('test_gen_age결측처리.xls')
#%%
'''
 gen_data 스케일링 
 '''
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

#%%
''' 
unu 전처리
'''

# In[388]:


# 결측치 종류 MAR, 개인 회생 납입 완료 여부 변수, 1을 제외하곤 개인회생조차 신청안했거나,
# 개인회생을 신청안했지만 납입 하지 않은 경우, 결측치 0으로 채워주면 됨
print('결측치 개수 파악')
train_unu['personal_rehabilitation_complete_yn'].fillna(0,inplace = True)
test_unu['personal_rehabilitation_complete_yn'].fillna(0,inplace = True)
print(f'개인회생납입 완료 여부 결측 개수: {train_unu.personal_rehabilitation_complete_yn.isnull().sum()}')
print(f'test개인회생납입 완료 여부 결측 개수: {test_unu.personal_rehabilitation_complete_yn.isnull().sum()}')


# In[389]:


#train셋의 user id로 탄생년도와 성별 채우기
# 두 변수는 개인의 고유적인 특징, 불변이기 때문에 채워줌

train_unu['birth_year'] = train_unu['birth_year'].fillna(train_unu.groupby('user_id')['birth_year'].transform('mean'))
train_unu['gender'] = train_unu['gender'].fillna(train_unu.groupby('user_id')['gender'].transform('mean'))

test_unu['birth_year'] = test_unu['birth_year'].fillna(train_unu.groupby('user_id')['birth_year'].transform('mean'))
test_unu['gender'] = test_unu['gender'].fillna(train_unu.groupby('user_id')['gender'].transform('mean'))


# In[390]:


# 나눠진 데이터 셋들의 정보에는, 개인 회생을 신청한 사람과, 신청하지 않은 사람의 정보가 담겨있음.
# 정보가 다 담겨 있기 때문에, 굳이 결측을 채워주지 않고, 변수를 제거하고 사용
# 개인회생 여부 포함하면서 나눠줌, gen, unu 셋에는 개인회생을 신청한 애들과, 신청안한 애들
# 나눠짐, 변수 제거해도 그 속성은 남아있어서 변수 제거
# unu는 개인회생 완납, 미납 차이있는지 살펴봐야해서 완납은 살려 둠
train_unu.drop(['personal_rehabilitation_yn'],axis=1,inplace =True)
test_unu.drop(['personal_rehabilitation_yn'],axis=1,inplace =True)

# ### train_unu 전처리

# In[424]:


train_unu.isnull().sum()


# In[425]:


# cnt 변수의 nan 값은 기대출 수가 없는 사람으로 판단하고 cat, amt 변수들의 nan값들을 다 0으로 채워줌
train_unu.loc[train_unu['existing_loan_cnt'] != train_unu['existing_loan_cnt'], 'existing_loan_cnt'] = 0
train_unu.loc[train_unu['existing_loan_amt'] != train_unu['existing_loan_amt'], 'existing_loan_amt'] = 0
# 기대출 수가 있지만 금융사에서 제공해주지 않아 amt변수의 값이 0인 것들은 기대출수의 따른 평균 값으로 대체
train_unu.loc[(train_unu['existing_loan_cnt'] == 1) & (train_unu['existing_loan_amt'] == 0),'existing_loan_amt'] = 14800757


# In[460]
# cnt 변수의 nan 값은 기대출 수가 없는 사람으로 판단하고 cat, amt 변수들의 nan값들을 다 0으로 채워줌
test_unu.loc[test_unu['existing_loan_cnt'] != test_unu['existing_loan_cnt'], 'existing_loan_cnt'] = 0
test_unu.loc[test_unu['existing_loan_amt'] != test_unu['existing_loan_amt'], 'existing_loan_amt'] = 0
# 기대출 수가 있지만 금융사에서 제공해주지 않아 amt변수의 값이 0인 것들은 기대출수의 따른 평균 값으로 대체
test_unu.loc[(test_unu['existing_loan_cnt'] == 1) & (test_unu['existing_loan_amt'] == 0),'existing_loan_amt'] = 14800757


# In[395]:


# train_unu 년월 분리
train_unu['입사_년도'] = train_unu['company_enter_month']//100
train_unu['입사_월'] = train_unu['company_enter_month']%100




# In[400]:


# test_unu 분리
test_unu_over = test_unu[test_unu['company_enter_month']>=202207]
test_unu_under = test_unu[test_unu['company_enter_month']<202207]
test_unu_null = test_unu[test_unu['company_enter_month'].isnull()==True]


# In[401]:


# 입사년월이 6자리인 애들
test_unu_under['입사_년도'] = test_unu_under['company_enter_month']//100
test_unu_under['입사_월'] = test_unu_under['company_enter_month']%100


# In[402]:


# 입사년월이 8자리인 애들
test_unu_over['입사_년도'] = test_unu_over['company_enter_month']//10000
test_unu_over['입사_월'] = (test_unu_over['company_enter_month']//100)%100


# In[403]:


# 각각 입사년월 나눠준 데이터 셋 합치기
test_unu_1 = pd.concat([test_unu_under, test_unu_over], axis = 0)
test_unu = pd.concat([test_unu_1, test_unu_null], axis = 0)

# In[405]:


train_unu['근속년도'] = 2022 - train_unu['입사_년도']  
train_unu['근속개월'] = train_unu['근속년도']*12 + train_unu['입사_월']

# In[407]:


test_unu['근속년도'] = 2022 - test_unu['입사_년도']  
test_unu['근속개월'] = test_unu['근속년도']*12 + test_unu['입사_월']


# In[408]:


train_unu['age'] = 2022 - train_unu['birth_year']  
test_unu['age'] = 2022 - test_unu['birth_year']  


# In[410]:


print(train_unu.isnull().sum())
print('\n')
print(test_unu.isnull().sum())


#%%
#결측치 처리

# 변수 유의성 확인을 위한 회귀분석, 및 독립변수로 문자로 들어가면 분산 분석 
for i in train_unu_loan.columns:
    model = ols('credit_score ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')


# In[46]:

#credit, personal_rehabilitation_complete_yn
train_unu_loan['credit_score'] = train_unu_loan['credit_score'].fillna(train_unu_drop_na.groupby(['credit_score', 

# In[47]:

for i in train_unu_loan.columns:
    model = ols('근속개월 ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
# In[48]:
#bank
train_unu_loan['근속개월'] = train_unu_loan['근속개월'].fillna(train_unu_drop_na.groupby(['bank_id'])['근속개월'].transform('mean'))


# In[49]:

train_unu_loan.isnull().sum()

# In[26]:
train_unu_enter.isnull().sum()

# In[28]:
train_unu_enter.drop(['근속개월','Unnamed: 0'],axis=1,inplace = True)
# In[32]:
train_unu_enter.isnull().sum()
# In[34]:

for i in train_unu_enter.columns:
    model = ols('credit_score ~ train_unu_enter[i]', train_unu_enter).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
# In[38]:

# loan_lim,
#bank,produ,loan_rate,is_applied,gender,employment,purpose,personal_rehabilitation_complete_yn,existing_loan_cnt
train_unu_enter['credit_score'] = train_unu_enter['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id','product_id',

# In[39]:

train_unu_enter.isnull().sum()
# In[29]:
train_unu_credit.isnull().sum()
# In[24]:
train_unu_credit.drop(['credit_score'],axis=1,inplace = True)

# In[16]:

train_unu_age.isnull().sum()

# In[17]:

train_unu_age.drop(['gender','age'],axis=1,inplace = True)

# In[18]:

for i in train_unu_age.columns:
    model = ols('근속개월 ~ train_unu_age[i]', train_unu_age).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')

# In[19]:

# bank_id,product,income_type(),employment_type,desired_amount,personal_rehabilitation_complete_yn,existing_loan_amt,
# loan_limit,rate,credit_score,houseown_type,purpose,existing_loan_cnt
train_unu_age['근속개월'] = train_unu_age['근속개월'].fillna(train_unu_drop_na.groupby(['loan_limit','loan_rate', 
                                                                                        'credit_score', 

# In[20]:


train_unu_age.isnull().sum()


# In[7]:



# In[21]:


# bank_id,product,income_type(),employment_type,desired_amount,personal_rehabilitation_complete_yn,existing_loan_amt,
# loan_limit,rate,credit_score,houseown_type,purpose,existing_loan_cnt
test_unu_age['근속개월'] = test_unu_age['근속개월'].fillna(train_unu_drop_na.groupby(['loan_limit','loan_rate', 

# In[22]:

test_unu_age.isnull().sum()

# In[25]:

test_unu_credit.isnull().sum()

# In[27]:

test_unu_enter.isnull().sum()

# In[40]:

test_unu_enter.drop(['근속개월'],axis=1,inplace = True)

# In[41]:

# loan_lim,
#bank,produ,loan_rate,is_applied,gender,employment,purpose,personal_rehabilitation_complete_yn,existing_loan_cnt
test_unu_enter['credit_score'] = test_unu_enter['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id','product_id',
                                                                                                    'loan_rate', 'is_applied',
# In[42]:

test_unu_enter.isnull().sum()

# In[50]:

test_unu_loan['근속개월'] = test_unu_loan['근속개월'].fillna(train_unu_drop_na.groupby(['bank_id'])['근속개월'].transform('mean'))

test_unu_loan['credit_score'] = test_unu_loan['credit_score'].fillna(train_unu_drop_na.groupby(['credit_score','personal_rehabilitation_complete_yn'])['credit_score'].transform('mean'))


# In[51]:

test_unu_loan.isnull().sum()

# In[52]:

test_unu_loan.drop(['loan_limit','loan_rate'],axis=1,inplace=True)


# In[54]:

for i in train_unu_loan.columns:
    model = ols('gender ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
# In[57]:

train_unu_loan.gender.value_counts()

# In[58]:

#
#credit, personal_rehabilitation_complete_yn
test_unu_loan['gender'] = test_unu_loan['gender'].fillna(1)


# In[55]:


for i in train_unu_loan.columns:
    model = ols('age ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')

# In[59]:


train_unu_loan.age.describe()


# In[60]:

sns.distplot(train_unu_loan.age)
plt.show()


# In[61]:

test_unu_loan.age.fillna(40,inplace = True)

# In[62]:

test_unu_loan.isnull().sum()

# In[411]:

# 모델링 과정, 변수 버리기, 수치형들만 놔두기
# 라벨 인코딩을 해야하므로 결국 범주형 변수들도 수치형으로 사용해야 함
# 변수 내 1,2,3,,,,200까지 있기 때문에 스케일링을 통해 그 영향을 줄임
# type 변수는 문자형 그대로, 더미변수로 생성
# 변수변환에 사용해준 변수도 버림
train_unu.drop(['Unnamed: 0.3','Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0',
                'application_id','birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)
# In[412]:
test_unu.drop(['Unnamed: 0.3','Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0',
               'application_id','birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)

# In[413]:


# 데이터 나눠주기
# 1단계 train_unu에서 loan 결측 행 분리
train_unu_copy = train_unu.copy()
train_unu_loan = train_unu_copy[train_unu_copy['loan_limit'].isnull()==True] # loan만 결측치인 행 추출
train_unu_drop_loan = train_unu_copy.dropna(subset=['loan_limit'],axis = 0) # loan결측치 제거된 데이터 셋


# In[414]:


# 2단계 train_unu에서 birth,gender 결측 행 분리
train_unu_age = train_unu_drop_loan[train_unu_drop_loan['age'].isnull()==True] # # loan결측치 제거된 데이터 셋에서 birth_year행 결측치 추출
train_unu_drop_age = train_unu_drop_loan.dropna(subset=['age'],axis = 0) # loan + birth_year 결측치가 존재하지 않는 데이터 셋


# In[415]:

# 3단계 train_unu에서 enter_month 결측 행 분리
train_unu_enter = train_unu_drop_age[train_unu_drop_age['근속개월'].isnull()==True] # company_enter_month만 결측치인 행 추출
train_unu_drop_enter = train_unu_drop_age.dropna(subset=['근속개월'],axis = 0) 
# loan + birth_year + company_enter_month결측치가 존재하지 않는 데이터 셋


# In[416]:

# 4단계 train_unu에서 credit_score결측 행 분리
train_unu_credit = train_unu_drop_enter[train_unu_drop_enter['credit_score'].isnull()==True] # credit_score만 결측치인 행 추출
train_unu_drop_na = train_unu_drop_enter.dropna(subset=['credit_score'],axis = 0) 
# loan + birth_year + company_enter_month + credit_score 결측치가 존재하지 않는 데이터 셋

# In[176]:

train_unu_age.to_csv('train_unu_age.csv')
train_unu_credit.to_csv('train_unu_credit.csv')
train_unu_drop_na.to_csv('train_unu_drop_na.csv')
train_unu_enter.to_csv('train_unu_enter.csv')
train_unu_loan.to_csv('train_unu_loan.csv')

# In[177]:

train_unu_age.to_csv('train_unu_age_cp.csv',encoding = 'cp949')
train_unu_credit.to_csv('train_unu_credit_cp.csv',encoding = 'cp949')
train_unu_drop_na.to_csv('train_unu_drop_na_cp.csv',encoding = 'cp949')
train_unu_enter.to_csv('train_unu_enter_cp.csv',encoding = 'cp949')
train_unu_loan.to_csv('train_unu_loan_cp.csv',encoding = 'cp949')

# In[413]:

# 데이터 나눠주기
# 1단계 test_unu에서 loan 결측 행 분리
test_unu_copy = test_unu.copy()
test_unu_loan = test_unu_copy[test_unu_copy['loan_limit'].isnull()==True] # loan만 결측치인 행 추출
test_unu_drop_loan = test_unu_copy.dropna(subset=['loan_limit'],axis = 0) # loan결측치 제거된 데이터 셋

# In[414]:

# 2단계 test_unu에서 birth,gender 결측 행 분리
test_unu_age = test_unu_drop_loan[test_unu_drop_loan['age'].isnull()==True] # # loan결측치 제거된 데이터 셋에서 birth_year행 결측치 추출
test_unu_drop_age = test_unu_drop_loan.dropna(subset=['age'],axis = 0) # loan + birth_year 결측치가 존재하지 않는 데이터 셋

# In[415]:

# 3단계 test_unu에서 enter_month 결측 행 분리
test_unu_enter = test_unu_drop_age[test_unu_drop_age['근속개월'].isnull()==True] # company_enter_month만 결측치인 행 추출
test_unu_drop_enter = test_unu_drop_age.dropna(subset=['근속개월'],axis = 0) 
# loan + birth_year + company_enter_month결측치가 존재하지 않는 데이터 셋

# In[416]:

# 4단계 test_unu에서 credit_score결측 행 분리
test_unu_credit = test_unu_drop_enter[test_unu_drop_enter['credit_score'].isnull()==True] # credit_score만 결측치인 행 추출
test_unu_drop_na = test_unu_drop_enter.dropna(subset=['credit_score'],axis = 0) 
# loan + birth_year + company_enter_month + credit_score 결측치가 존재하지 않는 데이터 셋

# In[176]:

test_unu_age.to_csv('test_unu_age.csv')
test_unu_credit.to_csv('test_unu_credit.csv')
test_unu_drop_na.to_csv('test_unu_drop_na.csv')
test_unu_enter.to_csv('test_unu_enter.csv')
test_unu_loan.to_csv('test_unu_loan.csv')

# In[177]:

test_unu_age.to_csv('test_unu_age_cp.csv',encoding = 'cp949')
test_unu_credit.to_csv('test_unu_credit_cp.csv',encoding = 'cp949')
test_unu_drop_na.to_csv('test_unu_drop_na_cp.csv',encoding = 'cp949')
test_unu_enter.to_csv('test_unu_enter_cp.csv',encoding = 'cp949')
test_unu_loan.to_csv('test_unu_loan_cp.csv',encoding = 'cp949')

# In[417]:

print(train_gen.columns)

# In[418]:

# train_unu 스케일링을 위한 수치형, 범주형 나누기
train_unu_num = train_unu_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]

# In[419]:

train_unu_ob = train_unu_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)

# In[]:
# test_unu 스케일링을 위한 수치형, 범주형 나누기
test_unu_num = test_unu_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]

# In[419]:

test_unu_ob = test_unu_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)

# In[ ]:

# unu 스케일링

# 이상치가 존재하므로 수치형 변수 gen 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
train_unu_scaled = rbs.fit_transform(train_unu_num) #fit시킨 데이터 적용
test_unu_scaled = rbs.transform(test_unu_num) #fit시킨 데이터 적용
#%%
#%%
train_unu_scaled = pd.DataFrame(data = train_unu_scaled)
test_unu_scaled = pd.DataFrame(data = test_unu_scaled)

#%%
train_unu_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']

test_unu_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']

# In[479]:

test_unu_num.isnull().sum()

# In[440]:

train_unu_age.to_csv('train_unu_age.csv')
train_unu_credit.to_csv('train_unu_credit.csv')
train_unu_drop_na.to_csv('train_unu_drop_na.csv')
train_unu_enter.to_csv('train_unu_enter.csv')
train_unu_loan.to_csv('train_unu_loan.csv')

# In[441]:

train_unu_age.to_csv('train_unu_age_cp.csv',encoding = 'cp949')
train_unu_credit.to_csv('train_unu_credit_cp.csv',encoding = 'cp949')
train_unu_drop_na.to_csv('train_unu_drop_na_cp.csv',encoding = 'cp949')
train_unu_enter.to_csv('train_unu_enter_cp.csv',encoding = 'cp949')
train_unu_loan.to_csv('train_unu_loan_cp.csv',encoding = 'cp949')

#%%
# train_unu 스케일링을 위한 수치형, 범주형 나누기
train_unu_num = train_unu_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
train_unu_ob = train_unu_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)

#%%
# test_unu 스케일링을 위한 수치형, 범주형 나누기
test_unu_num = test_unu_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_unu_ob = test_unu_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)

#%%
# 나눠준 데이터 합치기 concat
train_unu_sca = pd.concat([train_unu_ob,train_unu_scaled],axis=1)
test_unu_sca = pd.concat([test_unu_ob,test_unu_scaled],axis=1)

#%%
test_unu_sca.to_csv('test_unu_sca.csv')
#%%
test_unu_sca.to_csv('test_unu_sca_cp.csv',encoding = 'cp949')
#%%
train_unu_sca.to_csv('train_unu_sca.csv')
#%%
train_unu_sca.to_csv('train_unu_sca_cp.csv',encoding = 'cp949')

#%%
#수치형 변수만 남기기
#tr_gen_num = train_gen.copy().drop(['application_id', 'loanapply_insert_time', 'bank_id', 'product_id',
#       'user_id', 'birth_year','gender', 'insert_time','income_type',
#       'company_enter_month', 'employment_type', 'houseown_type',
#       'purpose', 'existing_loan_cnt','existing_loan_amt','month','Unnamed: 0'],axis=1)

tr_gen_num = train_unu_sca.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'gender', 'credit_score', 'yearly_income','desired_amount',
       'personal_rehabilitation_complete_yn', 'existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]

#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor
#다중공선성
tr_gen_num = tr_gen_num.dropna(axis=0)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(tr_gen_num.values, i) for i in range(tr_gen_num.shape[1])]
vif["features"] = tr_gen_num.columns
print(vif)
# 다중공선성 없음
#%%
# train_unu_age 스케일링을 위한 수치형, 범주형 나누기
train_unu_age_num = train_unu_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']]
#%%
train_unu_age_ob = train_unu_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월'],axis=1)

#%%
# test_unu_age 스케일링을 위한 수치형, 범주형 나누기
test_unu_age_num = test_unu_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']]
#%%
test_unu_age_ob = test_unu_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월'],axis=1)
#%%
# 이상치가 존재하므로 수치형 변수 unu_age 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_age) # 결측치 없는 train데이터들로 fit시키고
train_unu_age_scaled = rbs.fit_transform(train_unu_age_num) #fit시킨 데이터 적용
test_unu_age_scaled = rbs.transform(test_unu_age_num) #fit시킨 데이터 적용
#%%
# 배열 형태로 반환되므로, 데이터 프레임으로 변환
train_unu_age_scaled = pd.DataFrame(data = train_unu_age_scaled )
test_unu_age_scaled = pd.DataFrame(data = test_unu_age_scaled)
#%%
# 변수명 삽입
train_unu_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']
#%%
test_unu_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']
#%%
# 나눠준 데이터 합치기 concat
train_unu_age_sca = pd.concat([train_unu_age_ob,train_unu_age_scaled],axis=1)
test_unu_age_sca = pd.concat([test_unu_age_ob,test_unu_age_scaled],axis=1)
#%%
train_unu_age_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\train_unu_age_sca.csv',index=False)
test_unu_age_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\test_unu_age_sca.csv',index=False)


#%%
# train_unu_enter 스케일링을 위한 수치형, 범주형 나누기
train_unu_enter_num = train_unu_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'unuder','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']]
#%%
train_unu_enter_ob = train_unu_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'unuder','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age'],axis=1)

#%%
# test_unu_enter 스케일링을 위한 수치형, 범주형 나누기
test_unu_enter_num = test_unu_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'unuder','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']]
#%%
test_unu_enter_ob = test_unu_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'unuder','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age'],axis=1)
#%%
# 이상치가 존재하므로 수치형 변수 unu_enter 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_enter) # 결측치 없는 train데이터들로 fit시키고
train_unu_enter_scaled = rbs.fit_transform(train_unu_enter_num) #fit시킨 데이터 적용
test_unu_enter_scaled = rbs.transform(test_unu_enter_num) #fit시킨 데이터 적용
#%%
# 배열 형태로 반환되므로, 데이터 프레임으로 변환
train_unu_enter_scaled = pd.DataFrame(data = train_unu_enter_scaled )
test_unu_enter_scaled = pd.DataFrame(data = test_unu_enter_scaled)
#%%
# 변수명 삽입
train_unu_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'unuder','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']
#%%
test_unu_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'unuder','credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']
#%%
# 나눠준 데이터 합치기 concat
train_unu_enter_sca = pd.concat([train_unu_enter_ob,train_unu_enter_scaled],axis=1)
test_unu_enter_sca = pd.concat([test_unu_enter_ob,test_unu_enter_scaled],axis=1)
#%%
train_unu_enter_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\train_unu_enter_sca.csv',index=False)
test_unu_enter_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\test_unu_enter_sca.csv',index=False)

#%%
# train_unu_loan 스케일링을 위한 수치형, 범주형 나누기
train_unu_loan_num = train_unu_loan.copy()[['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
train_unu_loan_ob = train_unu_loan.copy().drop(['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)

#%%
# test_unu_loan 스케일링을 위한 수치형, 범주형 나누기
test_unu_loan_num = test_unu_loan.copy()[['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_unu_loan_ob = test_unu_loan.copy().drop(['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
#%%
# 이상치가 존재하므로 수치형 변수 unu_loan 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_loan) # 결측치 없는 train데이터들로 fit시키고
train_unu_loan_scaled = rbs.fit_transform(train_unu_loan_num) #fit시킨 데이터 적용
test_unu_loan_scaled = rbs.transform(test_unu_loan_num) #fit시킨 데이터 적용
#%%
# 배열 형태로 반환되므로, 데이터 프레임으로 변환
train_unu_loan_scaled = pd.DataFrame(data = train_unu_loan_scaled )
test_unu_loan_scaled = pd.DataFrame(data = test_unu_loan_scaled)
#%%
# 변수명 삽입
train_unu_loan_scaled.columns = ['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_unu_loan_scaled.columns = ['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
# 나눠준 데이터 합치기 concat
train_unu_loan_sca = pd.concat([train_unu_loan_ob,train_unu_loan_scaled],axis=1)
test_unu_loan_sca = pd.concat([test_unu_loan_ob,test_unu_loan_scaled],axis=1)
#%%
train_unu_loan_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\train_unu_loan_sca.csv',index=False)
test_unu_loan_sca.to_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\sca\\test_unu_loan_sca.csv',index=False)

#%%
# train_unu_age 스케일링을 위한 수치형, 범주형 나누기
train_unu_credit_num = train_unu_credit.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
        'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월','age']]
#%%
train_unu_credit_ob = train_unu_credit.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
        'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월','age'],axis=1)

#%%
# test_unu_age 스케일링을 위한 수치형, 범주형 나누기
test_unu_credit_num = test_unu_credit.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
        'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월','age']]
#%%
test_unu_credit_ob = test_unu_credit.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
        'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월','age'],axis=1)

# 이상치가 존재하므로 수치형 변수 unu_enter 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_enter) # 결측치 없는 train데이터들로 fit시키고
train_unu_credit_scaled = rbs.fit_transform(train_unu_credit_num) #fit시킨 데이터 적용
test_unu_credit_scaled = rbs.transform(test_unu_credit_num) #fit시킨 데이터 적용
#%%
# 배열 형태로 반환되므로, 데이터 프레임으로 변환
train_unu_credit_scaled = pd.DataFrame(data = train_unu_credit_scaled )
test_unu_credit_scaled = pd.DataFrame(data = test_unu_credit_scaled)
#%%
# 변수명 삽입
train_unu_credit_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
        'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt','근속개월', 'age']
#%%
test_unu_credit_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
        'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt','근속개월', 'age']
#%%
# 나눠준 데이터 합치기 concat
train_unu_credit_sca = pd.concat([train_unu_credit_ob,train_unu_credit_scaled],axis=1)
test_unu_credit_sca = pd.concat([test_unu_credit_ob,test_unu_credit_scaled],axis=1)

train_unu_credit_sca.to_csv('train_unu_credit_sca.csv',index=False)
test_unu_credit_sca.to_csv('test_unu_credit_sca.csv',index=False)

#%%
a = pd.read_csv('test_unu_drop_na')