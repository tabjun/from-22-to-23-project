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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
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
dest = r'C:\\Users\\215-01\\Desktop\\빅콘\\1014_빅콘 코드 총정리.ipynb'
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
train_gen.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)
#%%
train_unu.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)
#%%
test_gen.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)
#%%
test_unu.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)

#%%
'''
결측치행이 존재하는 변수별로 분리하여, 각 데이터 셋별 모델링
결측치를 채울 부담을 재워줌, 앙상블 기법
'''
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
train_gen_age.to_csv('train_gen_age_1013.csv',index = False)
train_gen_credit.to_csv('train_gen_credit_1013.csv',index = False)
train_gen_drop_na.to_csv('train_gen_drop_na_1013.csv',index = False)
train_gen_enter.to_csv('train_gen_enter_1013.csv',index = False)
train_gen_loan.to_csv('train_gen_loan_1013.csv',index = False)
#%%
train_gen_age.to_csv('train_gen_age_cp_1013.csv',encoding = 'cp949',index = False)
train_gen_credit.to_csv('train_gen_credit_cp_1013.csv',encoding = 'cp949',index = False)
train_gen_drop_na.to_csv('train_gen_drop_na_cp_1013.csv',encoding = 'cp949',index = False)
train_gen_enter.to_csv('train_gen_enter_cp_1013.csv',encoding = 'cp949',index = False)
train_gen_loan.to_csv('train_gen_loan_cp_1013.csv',encoding = 'cp949',index = False)
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
test_gen_age.to_csv('test_gen_age_1013.csv',index = False)
test_gen_credit.to_csv('test_gen_credit_1013.csv',index = False)
test_gen_drop_na.to_csv('test_gen_drop_na_1013.csv',index = False)
test_gen_enter.to_csv('test_gen_enter_1013.csv',index = False)
test_gen_loan.to_csv('test_gen_loan_1013.csv',index = False)
#%%
test_gen_age.to_csv('test_gen_age_cp_1013.csv',encoding = 'cp949',index = False)
test_gen_credit.to_csv('test_gen_credit_cp_1013.csv',encoding = 'cp949',index = False)
test_gen_drop_na.to_csv('test_gen_drop_na_cp_1013.csv',encoding = 'cp949',index = False)
test_gen_enter.to_csv('test_gen_enter_cp_1013.csv',encoding = 'cp949',index = False)
test_gen_loan.to_csv('test_gen_loan_cp_1013.csv',encoding = 'cp949',index = False)
#%%
# gen 데이터 셋 경로
os.chdir('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\gen_split 데이터 셋')
#%%
train_gen_drop_na=pd.read_csv('train_gen_drop_na_1013.csv')
test_gen_drop_na=pd.read_csv('test_gen_drop_na_1013.csv') 
#%%
train_gen_num = train_gen_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[419]:
train_gen_ob = train_gen_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
# In[]:
# test_gen 스케일링을 위한 수치형, 범주형 나누기
test_gen_num = test_gen_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[419]:
test_gen_ob = test_gen_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
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
# 배열 형태로 반환되므로, 데이터 프레임으로 변환
train_gen_scaled = pd.DataFrame(data = train_gen_scaled )
test_gen_scaled = pd.DataFrame(data = test_gen_scaled)
#%%
# 변수명 삽입
train_gen_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_gen_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
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
test_gen_sca = pd.concat([test_gen_ob,test_gen_scaled],axis=1)
#%%
train_gen_sca.to_csv('train_gen_drop_na_sca_cp_1013.csv',index=False,encoding='cp949')
train_gen_sca.to_csv('train_gen_drop_na_sca_1013.csv',index=False)
#%%
test_gen_sca.to_csv('test_gen_drop_na_sca_cp_1013.csv',index=False,encoding='cp949')
test_gen_sca.to_csv('test_gen_drop_na_sca_1013.csv',index=False)

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
#수치형 변수만 남기기
#tr_gen_num = train_gen.copy().drop(['application_id', 'loanapply_insert_time', 'bank_id', 'product_id',
#       'user_id', 'birth_year','gender', 'insert_time','income_type',
#       'company_enter_month', 'employment_type', 'houseown_type',
#       'purpose', 'existing_loan_cnt','existing_loan_amt','month','Unnamed: 0'],axis=1)

te_gen_num = test_gen_sca.copy()[['loan_limit','loan_rate','age','credit_score','yearly_income',
                                   'desired_amount','근속개월',
                                   'existing_loan_cnt','existing_loan_amt']]
#%%
#다중공선성
te_gen_num = te_gen_num.dropna(axis=0)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(te_gen_num.values, i) for i in range(tr_gen_num.shape[1])]
vif["features"] = tr_gen_num.columns
print(vif)
# 다중공선성 없음

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

# In[58]:
train_gen_loan.reset_index(drop = False, inplace = True)
# In[59]:
train_gen_drop_na.reset_index(drop = False, inplace = True)
# In[60]:
# 추출한 데이터이기에 index가 맞지 않음. reset_index로 index 재부여.
train_gen_age.reset_index(drop = False, inplace = True)
# In[61]:
train_gen_enter.reset_index(drop = False, inplace = True)
# In[124]:
test_gen_loan.reset_index(drop = False, inplace = True)
# In[125]:
test_gen_drop_na.reset_index(drop = False, inplace = True)
# In[126]:
test_gen_age.reset_index(drop = False, inplace = True)
# In[127]:
test_gen_enter.reset_index(drop = False, inplace = True)
# In[62]:
# gender는 최빈값으로 채워줌.
train_gen_loan['gender'] = train_gen_loan['gender'].fillna(1)
# In[96]:
test_gen_loan['gender'] = test_gen_loan['gender'].fillna(1)
# In[64]:
# 단변량으로 유의한 변수 선택 후 그룹화. 그룹별 평균으로 결측치 대체.
for i in train_gen_loan.columns:
    model = ols('credit_score ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월', 'age'
train_gen_loan['credit_score'] = train_gen_loan['credit_score'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월', 'age'])['credit_score'].transform('mean'))
# In[66]:
for i in train_gen_loan.columns:
    model = ols('근속개월 ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'
train_gen_loan['근속개월'] = train_gen_loan['근속개월'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['근속개월'].transform('mean'))
# In[68]:
for i in train_gen_loan.columns:
    model = ols('age ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt'
train_gen_loan['age'] = train_gen_loan['age'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt'])['age'].transform('mean'))
# In[29]:
# loan_limit, loan_rate가 결측치인 데이터 셋임. 두 변수 제거.
train_gen_loan = train_gen_loan.drop(['loan_limit', 'loan_rate'], axis = 1)
# In[31]:
# 결측치 없는 거 확인.
train_gen_loan.isnull().sum()
# In[136]:
for i in train_gen_age.columns:
    model = ols('credit_score ~ train_gen_age[i]', train_gen_age).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
test_gen_age['credit_score'] = test_gen_age['credit_score'].fillna(train_gen_drop_na.groupby(['bank_id', 'loan_rate', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월'])['credit_score'].transform('mean'))
# In[137]:
for i in train_gen_age.columns:
    model = ols('근속개월 ~ train_gen_age[i]', train_gen_age).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'loan_rate', 'employment_type', 'houseown_type', 'desired_amount',  'purpose', 'existing_loan_cnt'
test_gen_age['근속개월'] = test_gen_age['근속개월'].fillna(train_gen_drop_na.groupby(['loan_rate', 'employment_type', 'houseown_type', 'desired_amount',  'purpose', 'existing_loan_cnt'])['근속개월'].transform('mean'))
# In[140]:
# 결측치 없음을 확인.
test_gen_age.isnull().sum()
# In[141]:
# age, gender 변수 결측인 데이터 셋. 두 변수 제거.
test_gen_age = test_gen_age.drop(['gender', 'age'], axis = 1)
# test enter 결측치 처리
# In[143]:
for i in train_gen_enter.columns:
    model = ols('credit_score ~ train_gen_enter[i]', train_gen_enter).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'bank_id', 'product_id', 'loan_rate', 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'
test_gen_enter['credit_score'] = test_gen_enter['credit_score'].fillna(train_gen_drop_na.groupby(['bank_id', 'product_id', 'loan_rate', 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['credit_score'].transform('mean'))
# In[144]:
# 결측치 없음을 확인.
test_gen_enter.isnull().sum()
# In[145]:
# 근속개월 변수가 결측인 데이터 셋. 근속개월 변수 제거.
test_gen_enter = test_gen_enter.drop(['근속개월'], axis = 1)
# In[146]:
# 결측치가 없음을 확인.
test_gen_drop_na.isnull().sum()
#%%
# 스케일링 전 수치형과 문자형 데이터 분리.
train_gen_loan_num = train_gen_loan.copy()[['bank_id', 'product_id', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[193]:
train_gen_loan_ob = train_gen_loan.copy().drop(['bank_id', 'product_id', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'], axis = 1)
# In[195]:
test_gen_loan_num = test_gen_loan.copy()[['bank_id', 'product_id', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[196]:
test_gen_loan_ob = test_gen_loan.copy().drop(['bank_id', 'product_id', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'], axis = 1)
# In[197]:
train_gen_age_num = train_gen_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월']]
# In[198]:
train_gen_age_ob = train_gen_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월'], axis= 1)
# In[200]:
test_gen_age_num = test_gen_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월']]
# In[201]:
test_gen_age_ob = test_gen_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월'], axis= 1)
# In[202]:
train_gen_enter_num = train_gen_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', 'age']]
# In[203]:
train_gen_enter_ob = train_gen_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', 'age'], axis = 1)
# In[205]:
test_gen_enter_num = test_gen_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', 'age']]
# In[206]:
test_gen_enter_ob = test_gen_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', 'age'], axis = 1)
# In[207]:
train_gen_credit_num = train_gen_credit.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[208]:
train_gen_credit_ob = train_gen_credit.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'], axis= 1)
# In[209]:
test_gen_credit_num = test_gen_credit.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[210]:
test_gen_credit_ob = test_gen_credit.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'], axis= 1)
# In[211]:
train_gen_num = train_gen_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[212]:
train_gen_ob = train_gen_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
# In[213]:
test_gen_num = test_gen_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[214]:
test_gen_ob = test_gen_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
# In[216]:
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
# In[217]:
train_gen_loan_scaled = rbs.fit_transform(train_gen_loan_num) #fit시킨 데이터 적용
test_gen_loan_scaled = rbs.transform(test_gen_loan_num) #fit시킨 데이터 적용
# In[219]:
# np.array 형태를 DataFrame으로 변경.
train_gen_loan_scaled = pd.DataFrame(data = train_gen_loan_scaled )
test_gen_loan_scaled = pd.DataFrame(data = test_gen_loan_scaled)
# In[222]:
# 컬럼명 지정.
train_gen_loan_scaled.columns = ['bank_id', 'product_id', 'credit_score', 'yearly_income',
       'desired_amount', 'existing_loan_cnt', 'existing_loan_amt', '근속개월',
       'age']
# In[225]:
test_gen_loan_scaled.columns = ['bank_id', 'product_id', 'credit_score', 'yearly_income',
       'desired_amount', 'existing_loan_cnt', 'existing_loan_amt', '근속개월',
       'age']
# In[227]:
train_gen_age_scaled = rbs.fit_transform(train_gen_age_num) #fit시킨 데이터 적용
test_gen_age_scaled = rbs.transform(test_gen_age_num) #fit시킨 데이터 적용
# In[228]:
train_gen_age_scaled = pd.DataFrame(data = train_gen_age_scaled )
test_gen_age_scaled = pd.DataFrame(data = test_gen_age_scaled)
# In[230]:
train_gen_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', '근속개월']
# In[231]:
test_gen_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', '근속개월']
# In[232]:
train_gen_enter_scaled = rbs.fit_transform(train_gen_enter_num) #fit시킨 데이터 적용
test_gen_enter_scaled = rbs.transform(test_gen_enter_num) #fit시킨 데이터 적용
# In[233]:
train_gen_enter_scaled = pd.DataFrame(data = train_gen_enter_scaled )
test_gen_enter_scaled = pd.DataFrame(data = test_gen_enter_scaled)
# In[235]:
train_gen_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', 'age']
# In[236]:
test_gen_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', 'age']
# In[237]:
train_gen_credit_scaled = rbs.fit_transform(train_gen_credit_num) #fit시킨 데이터 적용
test_gen_credit_scaled = rbs.transform(test_gen_credit_num) #fit시킨 데이터 적용
# In[238]:
train_gen_credit_scaled = pd.DataFrame(data = train_gen_credit_scaled )
test_gen_credit_scaled = pd.DataFrame(data = test_gen_credit_scaled)
# In[240]:
train_gen_credit_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'yearly_income',
       'desired_amount', 'existing_loan_cnt', 'existing_loan_amt', '근속개월',
       'age']
# In[241]:
test_gen_credit_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'yearly_income',
       'desired_amount', 'existing_loan_cnt', 'existing_loan_amt', '근속개월',
       'age']

#%%
train_gen_drop_na_scaled = rbs.fit_transform(train_gen_num) #fit시킨 데이터 적용
test_gen_drop_na_scaled = rbs.transform(test_gen_num) #fit시킨 데이터 적용
# In[245]:
train_gen_drop_na_scaled = pd.DataFrame(data = train_gen_drop_na_scaled )
test_gen_drop_na_scaled = pd.DataFrame(data = test_gen_drop_na_scaled)
# In[247]:
train_gen_drop_na_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
# In[248]:
test_gen_drop_na_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
# In[250]:
# index가 맞지 않아 concat()을 사용할 수 없음. reset_index()로 재배치.
train_gen_loan_ob.reset_index(drop = False, inplace = True)
# In[251]:
train_gen_age_ob.reset_index(drop = False, inplace = True

# In[252]:
train_gen_enter_ob.reset_index(drop = False, inplace = True)
# In[253]:
train_gen_credit_ob.reset_index(drop = False, inplace = True)
# In[254]:
train_gen_ob.reset_index(drop = False, inplace = True)
# In[255]:
# concat() num과 ob를 합침.
train_gen_loan_sca = pd.concat([train_gen_loan_ob,train_gen_loan_scaled], axis=1)
# In[261]:
test_gen_loan_sca = pd.concat([test_gen_loan_ob,test_gen_loan_scaled], axis=1)
# In[273]:
# loan_limit, loan_rate가 결측치인 데이터. 두 변수 제거.
train_gen_loan_sca = train_gen_loan_sca.drop(['loan_limit', 'loan_rate'], axis = 1)
# In[258]:
train_gen_age_sca = pd.concat([train_gen_age_ob,train_gen_age_scaled], axis=1)
# In[264]:
test_gen_age_sca = pd.concat([test_gen_age_ob,test_gen_age_scaled], axis=1)
#%%
train_gen_enter_sca = pd.concat([train_gen_enter_ob,train_gen_enter_scaled], axis=1)
# In[266]:
test_gen_enter_sca = pd.concat([test_gen_enter_ob,test_gen_enter_scaled], axis=1)
# In[269]:
train_gen_credit_sca = pd.concat([train_gen_credit_ob,train_gen_credit_scaled], axis=1)
# In[270]:
test_gen_credit_sca = pd.concat([test_gen_credit_ob,test_gen_credit_scaled], axis=1)
# In[271]:
train_gen_drop_na_sca = pd.concat([train_gen_ob,train_gen_drop_na_scaled], axis=1)
# In[274]:
test_gen_drop_na_sca = pd.concat([test_gen_ob,test_gen_drop_na_scaled], axis=1)
# In[25]:
# credit_score가 결측인 데이트 셋. credit_score 변수 제거.
train_gen_credit = train_gen_credit.drop(['credit_score'], axis = 1)
# In[26]:
test_gen_credit = test_gen_credit.drop(['credit_score'], axis = 1)


#%%

train_gen_sca=pd.read_csv('train_gen_drop_na_sca.csv',encoding='cp949')
test_gen_sca=pd.read_csv('test_gen_drop_na_sca.csv',encoding='cp949')


# In[9]:


train_gen_sca.columns


# In[10]:


test_gen_sca.columns


# In[11]:


train_gen_sca.drop(['Unnamed: 0', 'level_0', 'index'], axis=1, inplace=True)


# In[13]:


test_gen_sca.drop(['Unnamed: 0', 'index'], axis=1, inplace=True)


# ## Sampling

# #### 1) 더미변수 

# In[20]:


train_gen_sca.columns


# In[21]:


final=pd.get_dummies(train_gen_sca, columns=['income_type', 'employment_type', 'houseown_type', 'purpose'])


# In[22]:


final.columns


# #### 2) sampling

# In[23]:


final.is_applied.value_counts()


# In[26]:


import seaborn as sns


# In[28]:


sns.countplot(x="is_applied", data=final)
plt.title('is_applied')
plt.show()


# # 불균형 데이터 비율 추출을 위한 층화추출

# In[29]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=777)
for train_idx, test_idx in split.split(final, final["is_applied"]):
    tr = final.loc[train_idx]
    val = final.loc[test_idx]


# In[30]:


print(tr["is_applied"].value_counts() / len(tr))
val["is_applied"].value_counts() / len(val)

# In[32]:


x_train=tr.drop(['is_applied','houseown_type_자가'], axis=1)
y_train=tr['is_applied']


# In[33]:


x_val=val.drop(['is_applied','houseown_type_자가'], axis=1)
y_val=val['is_applied']


# In[34]:


x_train.shape, x_val.shape


# - Grid Search 적용 X

# In[35]:


xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    )
#%%
xgb1.fit( x_train, y_train)


# In[36]:


pred = xgb1.predict(x_val)


# In[37]:


print(classification_report(y_val, pred, target_names=['class 0', 'class 1']))


# In[38]:


from sklearn.metrics import roc_auc_score
print('roc_auc_score {}'.format(roc_auc_score(y_val, pred)))


# In[39]:


from sklearn.metrics import roc_curve

pred_positive_label = xgb1.predict_proba(x_val)[:,1]

fprs, tprs, thresholds = roc_curve(y_val, pred_positive_label)

print('샘플 추츨')
print()

thr_idx = np.arange(1, thresholds.shape[0], 6)
print('thr idx:', thr_idx)
print('thr thresholds value:', thresholds[thr_idx])
print('thr thresholds value:', fprs[thr_idx])
print('thr thresholds value:', tprs[thr_idx])


# In[40]:


pred_positive_label = xgb1.predict_proba(x_val)[:,1]
fprs, tprs, thresholds = roc_curve(y_val, pred_positive_label)

precisions, recalls, thresholds = roc_curve(y_val, pred_positive_label)

plt.figure(figsize=(15, 5))

plt.plot([0,1], [0, 1], label='STR')

plt.plot(fprs, tprs, label='ROC')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid()
plt.show()


# ## test set 모델링

# In[41]:


test_gen_sca.isnull().sum()


# In[42]:

## 더미변수 생성
test=pd.get_dummies(test_gen_sca, columns=['income_type', 'employment_type', 'houseown_type', 'purpose'])


# In[43]:


test_x=test.drop(['is_applied','houseown_type_자가'], axis=1)


# In[48]:


xgb1_pred=xgb1.predict(test_x)
test_x.tail()


# In[49]:


a1 = pd.DataFrame(xgb1_pred)
a1.tail()


# In[50]:


a1.head()


# In[51]:

gen_drop_na = test_gen_sca.copy()
gen_drop_na['is_applied']=xgb1_pred
gen_drop_na.tail()


# In[52]:


gen_drop_na.to_csv('C:\\Users\\222-04\\Desktop\\gen_drop_na예측완_1014.csv')

# In[52]:

test.is_applied.value_counts()

# In[53]:

test.is_applied.value_counts()/len(test)

# In[27]:
# 모델링 전 범주형 변수들 더미변수로 변환.
final_loan = pd.get_dummies(train_gen_loan, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[29]:
# is_applied가 모두 1임. test set도 1로 채워줌.
(final_loan.is_applied.value_counts())
test_gen_loan['is_applied'] = test_gen_loan['is_applied'].fillna(1)
# In[96]:
final_age = pd.get_dummies(train_gen_age, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[99]:
X_train = final_age.drop(['Unnamed: 0', 'level_0', 'index','is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
y_train = final_age['is_applied']
# In[101]:
# is_applied가 0과 1의 비율이 20:1 정도로 편향이 심함. 오버샘플링 실시.
smote = SMOTE(random_state=42)
# In[102]:
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
# In[103]:
# XGBClassifier의 성능이 가장 좋게 나옴.
xgb_age = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    )
# In[104]:
xgb_age.fit( X_train, y_train)
# In[91]:
# ROC곡선으로 시각화 및 AUC 확인.
from sklearn.metrics import roc_curve
# In[92]:
pred_positive_label = xgb1.predict_proba(X_val)[:,1]
fprs, tprs, thresholds = roc_curve(y_val, pred_positive_label)
precisions, recalls, thresholds = roc_curve(y_val, pred_positive_label)
plt.figure(figsize=(15, 5))
plt.plot([0,1], [0, 1], label='STR')
plt.plot(fprs, tprs, label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid()
plt.show()
# In[111]:
# 사용하지 않는 데이터 제거.
test_gen_age = test_gen_age.drop(['Unnamed: 0', 'index'], axis = 1)
# In[112]:
dum_test_age = pd.get_dummies(test_gen_age,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[113]:
x_test = dum_test_age.drop(['is_applied','houseown_type_자가', 'income_type_PRIVATEBUSINESS'],axis=1)
# In[115]:
test_age_predict = xgb_age.predict(x_test)
# In[116]:
# is_applied 예측값 데이터프레임으로 변경.
a1= pd.DataFrame(test_age_predict)
# In[117]:
test_gen_age.is_applied = a1
# In[118]:
test_gen_age.to_csv('test_gen_age예측완료1_cp.csv',encoding = 'cp949')
# In[119]:
# 같은 방법으로 나머지 데이터 예측.
final_enter = pd.get_dummies(train_gen_enter, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[120]:
(final_enter.is_applied.value_counts())
# In[122]:
X_train = final_enter.drop(['Unnamed: 0', 'level_0', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
y_train = final_enter['is_applied']
# In[130]:
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
# In[131]:
xgb_enter.fit( X_train, y_train)
# In[132]:
dum_test_enter = pd.get_dummies(test_gen_enter,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[133]:
x_test = dum_test_enter.drop(['Unnamed: 0', 'index', 'is_applied','houseown_type_자가', 'houseown_type_배우자'],axis=1)
# In[134]:
X_train.columns
# In[135]:
x_test.columns
# In[136]:
test_enter_predict = xgb_enter.predict(x_test)
# In[137]:
a1= pd.DataFrame(test_enter_predict)
# In[138]:
test_gen_enter.is_applied = a1
# In[140]:
test_gen_enter.to_csv('test_gen_enter예측완료1_cp.csv',encoding = 'cp949')
# In[142]:
# credit 예측
final_credit = pd.get_dummies(train_gen_credit, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[143]:
(final_credit.is_applied.value_counts())
# In[145]:
X_train = final_credit.drop(['Unnamed: 0', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
y_train = final_credit['is_applied']
# In[146]:
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
# In[147]:
xgb_credit.fit( X_train, y_train)
# In[153]:
dum_test_credit = pd.get_dummies(test_gen_credit,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[154]:
x_test = dum_test_credit.drop(['Unnamed: 0', 'Unnamed: 0.1', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
# In[157]:
test_credit_predict = xgb_credit.predict(x_test)
# In[158]:
a1= pd.DataFrame(test_credit_predict)
# In[159]:
test_gen_credit.is_applied = a1
# In[160]:
test_gen_credit.isnull().sum()
# In[161]:
test_gen_credit.to_csv('test_gen_credit예측완료1_cp.csv',encoding = 'cp949')
# In[162]:
# drop_na 예측
final_drop_na = pd.get_dummies(train_gen_drop_na, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[163]:
(final_drop_na.is_applied.value_counts())
# In[164]:
X_train = final_drop_na.drop(['Unnamed: 0', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
y_train = final_drop_na['is_applied']
# In[165]:
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
# In[ ]:
xgb_drop_na.fit( X_train, y_train)
# In[ ]:
dum_test_drop_na = pd.get_dummies(test_gen_drop_na,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[ ]:
x_test = dum_test_drop_na.drop(['Unnamed: 0', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
# In[ ]:
X_train.columns
# In[ ]:
x_test.columns
# In[ ]:
test_drop_na_predict = xgb1.predict(x_test)
# In[ ]:
a1= pd.DataFrame(test_drop_na_predict) 
# In[ ]:
test_gen_drop_na.is_applied = a1
# In[ ]:
test_gen_drop_na.isnull().sum()
# In[ ]:
test_gen_drop_na.to_csv('test_gen_drop_na예측완료1_cp.csv',encoding = 'cp949')








#%%
''' 
unu 전처리
'''
os.chdir('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908')
#%%
train_unu = pd.read_csv('train_unu_1013.csv',encoding='cp949')
test_unu = pd.read_csv('test_unu_1013.csv',encoding='cp949')


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

#%%
train_unu.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)
#%%
test_unu.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)  
# In[410]:
print(train_unu.isnull().sum())
print('\n')
print(test_unu.isnull().sum())

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

#%%

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

#%%
train_unu_age.to_csv('train_unu_age_1013.csv',index=False)
train_unu_loan.to_csv('train_unu_loan_1013.csv',index=False)
train_unu_enter.to_csv('train_unu_enter_1013.csv',index=False)
train_unu_drop_na.to_csv('train_unu_drop_na_1013.csv',index=False)
train_unu_credit.to_csv('train_unu_credit_1013.csv',index=False)

#%%
test_unu_age.to_csv('test_unu_age_1013.csv',index=False)
test_unu_loan.to_csv('test_unu_loan_1013.csv',index=False)
test_unu_enter.to_csv('test_unu_enter_1013.csv',index=False)
test_unu_drop_na.to_csv('test_unu_drop_na_1013.csv',index=False)
test_unu_credit.to_csv('test_unu_credit_1013.csv',index=False)

#%%
# 스케일링 과정에 결측치가 포함되면, 스케일링 결과가 모두 결측치로 나옴
# 올바른 스케일링을 위한, 결측치 제거
#%%
# loan 데이터 전처리
print(train_unu_loan.isnull().sum())
'''
단변량 회귀분석을 통해 파악
이 때 p_val가 5% 유의수준에서 귀무가설을 기각, 즉 종속변수에 영향을 준다고 나와도,
절편도 같이 파악
절편이 0.00자리면 변수에 영향을 주지 않는다고 임의 판단하고 변수에서 제외
'''
#%%
# loan데이터의 credit_score 결측치 처리를 위한 단변량 회귀분석
# kcb신용점수 기준에서 기대출수가 영향을 준다고 나오기 때문에, 기대출 수는 groupby에 포함
for i in train_unu_loan.columns:
    model = ols('credit_score ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
    
#%%
#credit, personal_rehabilitation_complete_yn, 단변량 결과 얘들이 유의함
train_unu_loan['credit_score'] = train_unu_loan['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id', 
                                                                                                  'personal_rehabilitation_complete_yn',
                                                                                                 'existing_loan_cnt'])['credit_score'].transform('mean'))
#%%
# credit_score 잘 채워짐
print(train_unu_loan.isnull().sum())
#%%
# 근속개월 결측치 채우기 위한 단변량 회귀분석
for i in train_unu_loan.columns:
    model = ols('근속개월 ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
    
#%%
#bank_id만 유의
train_unu_loan['근속개월'] = train_unu_loan['근속개월'].fillna(train_unu_drop_na.groupby(['bank_id'])['근속개월'].transform('mean'))

#%%
print(train_unu_enter.isnull().sum())
# credit_score, 근속개월 결측치 확인
#%%
# 근속 개월결측치로 이뤄진 데이터 셋, 근속개월 변수 삭제
train_unu_enter.drop(['근속개월'],axis=1,inplace = True)
#%%
# enter데이터 셋의 credit_score
for i in train_unu_enter.columns:
    model = ols('credit_score ~ train_unu_enter[i]', train_unu_enter).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
    
#%%
#bank,produ,loan_rate,is_applied,gender,employment,purpose,personal_rehabilitation_complete_yn,existing_loan_cnt 유의
train_unu_enter['credit_score'] = train_unu_enter['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id','product_id',
                                                                                                    'loan_rate', 'is_applied',
                                                                                                    'purpose','gender',
                                                                                                    'personal_rehabilitation_complete_yn',
                                                                                                    'existing_loan_cnt'])['credit_score'].transform('mean'))

#%%
# 결측치 모두 처리됨을 확인
print(train_unu_enter.isnull().sum())
#%%
# credit 데이터셋 처리준비
print(train_unu_credit.isnull().sum())
#%%
# credit결측 데이터셋이므로 변수 제거
train_unu_credit.drop(['credit_score'],axis=1,inplace = True)
#%%
# age변수 처리
print(train_unu_age.isnull().sum())
#%%
# gender와age 같은 행에 존재하므로, 같이 제거
train_unu_age.drop(['gender','age'],axis=1,inplace = True)
#%%
# 결측치 존재하지 않음
print(train_unu_credit.isnull().sum())
#%%
#age 데이터셋의 근속개월 결측치 처리를 위한, 단변량 회귀분석
for i in train_unu_age.columns:
    model = ols('근속개월 ~ train_unu_age[i]', train_unu_age).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
#%%
# loan_limit,rate,credit_score,houseown_type,purpose,existing_loan_cnt 유의함
train_unu_age['근속개월'] = train_unu_age['근속개월'].fillna(train_unu_drop_na.groupby(['loan_limit','loan_rate', 
                                                                                                'credit_score', 
                                                                                                'houseown_type', 'purpose',
                                                                                                'existing_loan_cnt'])['근속개월'].transform('mean'))

#%%
# 결측치 다 채워짐을 확인
print(train_unu_age.isnull().sum())
#%%
'''
test_unu split 셋 처리
train_unu셋과 같이 처리
해당 데이터셋의 모두 결측행 변수 처리 및 단변량을 이용한 변수 groupby 채우기
'''
#%%

test_unu_age.drop(['birth_year','gender','age'],axis=1,inplace=True)
#%%
# loan_limit,rate,credit_score,houseown_type,purpose,existing_loan_cnt
test_unu_age['근속개월'] = test_unu_age['근속개월'].fillna(train_unu_drop_na.groupby(['loan_limit','loan_rate', 
                                                                                                'credit_score', 
                                                                                                'houseown_type', 'purpose',
                                                                                                'existing_loan_cnt'])['근속개월'].transform('mean'))

#%%
# loan_limit,rate,houseown_type,purpose,existing_loan_cnt
test_unu_age['credit_score'] = test_unu_age['credit_score'].fillna(train_unu_drop_na.groupby(['existing_loan_cnt'])['근속개월'].transform('mean'))
#%%
# 결측치 처리완
print(test_unu_age.isnull().sum())
#%%
#credit결측치 확인
print(test_unu_credit.isnull().sum())

#%%
# enter결측치 확인
print(test_unu_enter.isnull().sum())

#%%
test_unu_enter.drop(['근속개월'],axis=1,inplace = True)
#%%
#bank,produ,loan_rate,is_applied,gender,employment,purpose,personal_rehabilitation_complete_yn,existing_loan_cnt
test_unu_enter['credit_score'] = test_unu_enter['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id','product_id',
                                                                                                    'loan_rate', 'is_applied',
                                                                                                    'purpose','gender','employment_type',
                                                                                                    'personal_rehabilitation_complete_yn',
                                                                                                    'existing_loan_cnt'])['credit_score'].transform('mean'))
#%%
print(test_unu_enter.isnull().sum())
#%%
test_unu_loan['credit_score'] = test_unu_loan['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id', 
                                                                                                  'personal_rehabilitation_complete_yn',
                                                                                                 'existing_loan_cnt'])['credit_score'].transform('mean'))
#%%
test_unu_loan['근속개월'] = test_unu_loan['근속개월'].fillna(train_unu_drop_na.groupby(['bank_id'])['근속개월'].transform('mean'))
#%%
test_unu_loan.drop(['loan_rate','loan_limit'],axis=1,inplace=True)
#%%
test_unu_loan.isnull().sum()
#%%
for i in train_unu_loan.columns:
    model = ols('gender ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
    
#%%
print(train_unu_loan.gender.value_counts())
#%%
# 최빈값으로 채움
test_unu_loan['gender'] = test_unu_loan['gender'].fillna(1)
#%%
for i in train_unu_loan.columns:
    model = ols('age ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
#%%
print(train_unu_loan.age.describe())
#%%
sns.distplot(train_unu_loan.age)
plt.show()
#%%
test_unu_loan.age.fillna(40,inplace = True)
#%%
print(test_unu_loan.isnull().sum())
#%%

'''                     스케일링 작업                 '''

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
# 스케일링 전 결측치 확인
print(train_unu_age.isnull().sum())
print(test_unu_age.isnull().sum())
#%%
# 이상치가 존재하므로 수치형 변수 unu_age 로버스트 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_age) # 결측치 없는 train데이터들로 fit시키고
train_unu_age_scaled = rbs.fit_transform(train_unu_age_num) #fit시킨 데이터 적용
test_unu_age_scaled = rbs.transform(test_unu_age_num) #fit시킨 데이터 적용
#%%
# 스케일링 해주면 numpy배열로 나옴, 데이터프레임화해줘야함
train_unu_age_scaled = pd.DataFrame(data = train_unu_age_scaled )
test_unu_age_scaled = pd.DataFrame(data = test_unu_age_scaled)
#%%
# 변수명 넣기
train_unu_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']
#%%
test_unu_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']
#%%
# train_gen_ob셋은 원래 데이터 셋에서 행들을 제거해준것이기 때문에 인덱스가 일정하지 않음
# train_gen_scaled는 새로 추출해서 한 값이기에 인덱스가 1~800000까지 일정
train_unu_age_scaled.reset_index(drop = False, inplace = True)
train_unu_age_ob.reset_index(drop = False, inplace = True)

test_unu_age_scaled.reset_index(drop = False, inplace = True)
test_unu_age_ob.reset_index(drop = False, inplace = True)
#%%
train_unu_age_sca = pd.concat([train_unu_age_ob,train_unu_age_scaled],axis=1)
test_unu_age_sca = pd.concat([test_unu_age_ob,test_unu_age_scaled],axis=1)
#%%
print(train_unu_age_sca.shape)
print(train_unu_age.shape)
#%%
# enter 스케일링
train_unu_enter_num = train_unu_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']]
#%%
train_unu_enter_ob = train_unu_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age'],axis=1)
#%%
test_unu_enter_num = test_unu_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']]
#%%
test_unu_enter_ob = test_unu_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age'],axis=1)
#%%
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_enter) # 결측치 없는 train데이터들로 fit시키고
train_unu_enter_scaled = rbs.fit_transform(train_unu_enter_num) #fit시킨 데이터 적용
test_unu_enter_scaled = rbs.transform(test_unu_enter_num) #fit시킨 데이터 적용
#%%
train_unu_enter_scaled = pd.DataFrame(data = train_unu_enter_scaled )
test_unu_enter_scaled = pd.DataFrame(data = test_unu_enter_scaled)
#%%
# 변수명 삽입
train_unu_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']
#%%
test_unu_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']
#%%
train_unu_enter_scaled.reset_index(drop = False, inplace = True)
train_unu_enter_ob.reset_index(drop = False, inplace = True)

test_unu_enter_scaled.reset_index(drop = False, inplace = True)
test_unu_enter_ob.reset_index(drop = False, inplace = True)
#%%
train_unu_enter_sca = pd.concat([train_unu_enter_ob,train_unu_enter_scaled],axis=1)
test_unu_enter_sca = pd.concat([test_unu_enter_ob,test_unu_enter_scaled],axis=1)
#%%
print(train_unu_enter_sca.shape)
print(train_unu_enter.shape)
#%%
# loan
train_unu_loan_num = train_unu_loan.copy()[['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
train_unu_loan_ob = train_unu_loan.copy().drop(['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
#%%
test_unu_loan_num = test_unu_loan.copy()[['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_unu_loan_ob = test_unu_loan.copy().drop(['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
#%%
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_loan) # 결측치 없는 train데이터들로 fit시키고
train_unu_loan_scaled = rbs.fit_transform(train_unu_loan_num) #fit시킨 데이터 적용
test_unu_loan_scaled = rbs.transform(test_unu_loan_num) #fit시킨 데이터 적용
#%%
train_unu_loan_scaled = pd.DataFrame(data = train_unu_loan_scaled )
test_unu_loan_scaled = pd.DataFrame(data = test_unu_loan_scaled)
#%%
train_unu_loan_scaled.columns = ['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_unu_loan_scaled.columns = ['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
# train_gen_ob셋은 원래 데이터 셋에서 행들을 제거해준것이기 때문에 인덱스가 일정하지 않음
# train_gen_scaled는 새로 추출해서 한 값이기에 인덱스가 1~800000까지 일정
train_unu_loan_scaled.reset_index(drop = False, inplace = True)
train_unu_loan_ob.reset_index(drop = False, inplace = True)

test_unu_loan_scaled.reset_index(drop = False, inplace = True)
test_unu_loan_ob.reset_index(drop = False, inplace = True)
#%%
train_unu_loan_sca = pd.concat([train_unu_loan_ob,train_unu_loan_scaled],axis=1)
test_unu_loan_sca = pd.concat([test_unu_loan_ob,test_unu_loan_scaled],axis=1)
#%%
print(train_unu_loan_sca.shape)
print(train_unu_loan.shape)
#%%
# train_drop_na 스케일링을 위한 수치형, 범주형 나누기
# credit test셋은 is_applied외에 다른 변수에 결측치가 존재하지 않기 때문에
# drop_na데이터로 스케일링 및 train모델로 예측 적용
train_unu_num = train_unu_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
train_unu_ob = train_unu_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
#%%
# test_drop_na 스케일링을 위한 수치형, 범주형 나누기
test_unu_num = test_unu_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_unu_ob = test_unu_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
#%%
# unu_credit 스케일링을 위한 수치형, 범주형 나누기
test_unu_credit_num = test_unu_credit.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_unu_credit_ob = test_unu_credit.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)

#%%
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_gen) # 결측치 없는 train데이터들로 fit시키고
train_unu_scaled = rbs.fit_transform(train_unu_num) #fit시킨 데이터 적용
test_unu_credit_scaled = rbs.transform(test_unu_credit_num) #fit시킨 데이터 적용
#%%
train_unu_scaled = pd.DataFrame(data = train_unu_scaled)
test_unu_credit_scaled = pd.DataFrame(data = test_unu_credit_scaled)
test_unu_scaled = pd.DataFrame(data = test_unu_scaled)
#%%
train_unu_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_unu_credit_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_unu_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
train_unu_scaled.reset_index(drop = False, inplace = True)
train_unu_ob.reset_index(drop = False, inplace = True)

test_unu_scaled.reset_index(drop = False, inplace = True)
test_unu_ob.reset_index(drop = False, inplace = True)
#%%
test_unu_credit_sca = pd.concat([test_unu_credit_ob,test_unu_credit_scaled],axis=1)
#%%
train_unu_scaled.reset_index(drop = False, inplace = True)
train_unu_ob.reset_index(drop = False, inplace = True)

test_unu_scaled.reset_index(drop = False, inplace = True)
test_unu_ob.reset_index(drop = False, inplace = True)
#%%
train_unu_sca = pd.concat([train_unu_ob,train_unu_scaled],axis=1)
test_unu_sca = pd.concat([test_unu_ob,test_unu_scaled],axis=1)
#%%
print(test_unu_sca.shape)
print(test_unu_drop_na.shape)

#%%

'''                     prediction                ''' 


#%%

''' drop_na  prediction'''
# is_applied의 비율을 그대로 유지한 채 표본을 추출하기 위해 층화추출법으로 데이터를
# 나누어 줌
final_drop_na = pd.get_dummies(train_unu_sca,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# drop_na
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=777)
for train_idx, test_idx in split.split(final_drop_na, final_drop_na["is_applied"]):
    tr = final_drop_na.loc[train_idx]
    val = final_drop_na.loc[test_idx]

#%%
x_train=tr.drop(['is_applied','houseown_type_자가', 'application_id','level_0','index'], axis=1)
y_train=tr['is_applied']

x_val=val.drop(['is_applied','houseown_type_자가', 'application_id','level_0','index'], axis=1)
y_val=val['is_applied']
#%%
# 오버 샘플링
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(x_train, y_train)
print("SMOTE 적용 전 학습용 피처/레이블 데이터 세트 : ", x_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트 :', X_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 값의 분포 :\n',pd.Series(y_train_over).value_counts() )
#%%
# drop_na 모델링
xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    )
#%%
xgb1.fit(X_train_over, y_train_over)
pred = xgb1.predict(x_val)
print(classification_report(y_val, pred, target_names=['class 0', 'class 1']))
#%%
xgb1_pred=xgb1.predict(test_x)
xgb1_pred
#%%
unu_drop_na = test_unu_drop_na_sca.copy()
unu_drop_na['is_applied']=xgb1_pred
unu_drop_na.tail()
#%%
unu_drop_na.to_csv('unu_drop_na_xg_예측완_1013.csv')
#%%

#%%
xgb_credit = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    )
#%%
cr_train = X_train_over.drop(['purpose_자동차구입','purpose_주택구입'],axis=1)
#%%
xgb_credit.fit(cr_train, y_train_over)
#%%
unu_credit = test_unu_credit_sca.copy()
unu_credit['is_applied']=xgb_cr_pred
unu_credit.tail()
#%%
unu_credit.to_csv('unu_credit_예측완_1013_제대로.csv',index=False)


#%%
'''   loan prediction  '''
final_loan = pd.get_dummies(train_unu_loan_sca,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])

loan_x=final_loan.drop(['is_applied','houseown_type_자가', 'application_id'], axis=1)
loan_y=final_loan['is_applied']

#%%
xgb_loan = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    )

#%%
# loan 데이터 셋은, train set을 보았을 때 모두 1인 애들
# 예측 불가능, loan 결측행으로 분리한 데이터 셋의 정보는 모두 대출을 신청한 사람으로
# 간주하고 fillna(1)통해 is_applied를 채워줌
xgb_loan.fit(x_train, y_train)
#%%
test_unu_loan.is_applied.fillna(1)

#%%
 ''' age prediction   '''
final_age = pd.get_dummies(train_unu_age_sca,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])

test_age = pd.get_dummies(test_unu_age_sca,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
#%%
age_x=final_age.drop(['index','income_type_OTHERINCOME','application_id',
                      'employment_type_계약직','is_applied','houseown_type_자가', 
                      'application_id'], axis=1)
age_y=final_age['is_applied']

test_age.drop(['income_type_PRIVATEBUSINESS','purpose_전월세보증금', 'purpose_주택구입','purpose_생활비',
               'index','is_applied','employment_type_일용직','houseown_type_자가','purpose_사업자금','application_id'
              ],axis=1,inplace=True)
#%%
age_x=final_age.drop(['index','income_type_OTHERINCOME','application_id',
                      'employment_type_계약직','is_applied','houseown_type_자가', 
                      'application_id'], axis=1)
age_y=final_age['is_applied']
#%%

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_age_over, y_age_over = smote.fit_resample(age_x, age_y)
print("SMOTE 적용 전 학습용 피처/레이블 데이터 세트 : ", age_x.shape, age_y.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트 :', x_age_over.shape, y_age_over.shape)
print('SMOTE 적용 후 값의 분포 :\n',pd.Series(y_age_over).value_counts() )
#%%
xgb_age = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    )
#%%
xgb_age.fit(x_age_over,y_age_over)
#%%
age_pred = xgb_age.predict(test_age)
#%%
unu_age = test_unu_age_sca.copy()
unu_age.is_applied = age_pred
unu_age.is_applied.value_counts()
#%%
unu_age.to_csv('unu_age_예측진짜완_1013.csv')

#%%
''' enter prediction '''
final_enter = pd.get_dummies(train_unu_enter_sca,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# loan
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=777)
for train_idx, test_idx in split.split(final_enter, final_enter["is_applied"]):
    tr_enter = final_enter.loc[train_idx]
    val_enter = final_enter.loc[test_idx]

#%%
x_enter=tr_enter.drop(['is_applied','houseown_type_자가', 'employment_type_계약직','application_id','index'], axis=1)
y_enter=tr_enter['is_applied']

x_val=val.drop(['is_applied','houseown_type_자가', 'application_id','index'], axis=1)
y_val=val['is_applied']
#%%
# 오버 샘플링
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_enter_ov, y_enter_ov = smote.fit_resample(x_enter, y_enter)
print("SMOTE 적용 전 학습용 피처/레이블 데이터 세트 : ", x_enter.shape, y_enter.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트 :', x_enter_ov.shape, y_enter_ov.shape)
print('SMOTE 적용 후 값의 분포 :\n',pd.Series(y_enter_ov).value_counts())

#%%
xgb_enter = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=-1,
    scale_pos_weight=1,
    )
#%%
xgb_enter.fit(x_enter_ov,y_enter_ov)
pred = xgb_enter.predict(x_val)
print(classification_report(y_val, pred, target_names=['class 0', 'class 1']))

#%%
## 더미변수 생성
test_enter=pd.get_dummies(test_unu_enter_sca, columns=['income_type', 'employment_type', 'houseown_type', 'purpose'])
#%%
test_enter=test_enter.drop(['index','birth_year','is_applied','application_id'], axis=1)
test_enter=test_enter.drop(['employment_type_정규직','houseown_type_자가'], axis=1)
#%%
enter_pred= xgb_enter.predict(test_enter)
#%%
unu_enter = test_unu_enter_sca.copy()
unu_enter['is_applied'] = enter_pred
unu_enter
#%%
unu_enter.to_csv('unu_enter_예측완_1013.csv',index=False)

#%%
age = pd.read_csv('unu_age_예측진짜완_1013.csv')
credit = pd.read_csv('unu_credit_예측완_1013_제대로.csv')
nona = pd.read_csv('unu_drop_na_xg_예측완_1013.csv')
loan = pd.read_csv('unu_loan_예측완_1013.csv')
enter = pd.read_csv('unu_enter_예측완_1013.csv')
#%%
age_c = age[['application_id','product_id','is_applied']]
credit_c = credit[['application_id','product_id','is_applied']]
nona_c = nona[['application_id','product_id','is_applied']]
loan_c = loan[['application_id','product_id','is_applied']]
enter_c = enter[['application_id','product_id','is_applied']]
#%%
print(loan_c.isnull().sum())
#%%
# loan결측치, 이거는 그냥 결측치 1로 채워도됨
#%%
loan_c.is_applied.fillna(1,inplace=True)
#%%
unu_result = pd.concat([age_c,credit_c,nona_c,loan_c,enter_c],axis=0)
print(unu_result.isnull().sum())
#%%
unu_result.to_csv('unu예측정리완료.csv',index=False)
unu_result.to_csv('unu예측정리완료_cp.csv',index=False,encoding='cp949')
#%%
test_unu = pd.read_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\test_unu_1013.csv',encoding='cp949')
#%%
unu_result.sort_values('application_id',axis=1)
#%%
unu_result.sort_values(by='application_id' ,ascending=True,inplace=True)
#%%
test_unu.sort_values(by='application_id' ,ascending=True,inplace=True)
#%%
unu_result.drop(['product_id'],axis=1,inplace=True)
#%%
unu_result.reset_index(drop = False, inplace = True)
test_unu.reset_index(drop = False, inplace = True)

#%%
unu_result['product_id'] = test_unu['product_id']
#%%
unu_result = unu_result[['application_id','product_id','is_applied']]
#%%
unu_result.to_csv('unu예측완료 및 product_id완료.csv',index=False)

# In[ ]:


test_gen_loan_c = test_gen_loan[['application_id','product_id','is_applied']]


# In[ ]:


test_gen_age_c = test_gen_age[['application_id','product_id','is_applied']]


# In[ ]:


test_gen_enter_c = test_gen_enter[['application_id','product_id','is_applied']]


# In[ ]:


test_gen_credit_c = test_gen_credit[['application_id','product_id','is_applied']]


# In[ ]:


gen = pd.concat([test_gen_loan_c,test_gen_age_c,test_gen_enter_c,test_gen_credit_c, test_gen_drop_na_c],axis=0)


# In[ ]:


gen.drop(['product_id'],axis=1,inplace=True)


# In[ ]:


#%%
gen.sort_values(by='application_id' ,ascending=True,inplace=True)
#%%
test_gen.sort_values(by='application_id' ,ascending=True,inplace=True)


# In[ ]:


gen.reset_index(drop = False, inplace = True)


# In[ ]:


test_gen.reset_index(drop = False, inplace = True)


# In[ ]:


gen['product_id'] = test_gen['product_id']


# In[ ]:


gen = gen.drop(['index'], axis = 1)

#%%
gen_pred = pd.read_csv('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908\\gen예측결과.csv')
#%%
total_pred = pd.concat([gen_pred,unu_result])
#%%
total_pred.drop(['Unnamed: 0'],axis=1,inplace = True)
#%%
total_pred.to_csv('데이터분석분야_퓨처스뷰문_평가데이터.csv',index=False)