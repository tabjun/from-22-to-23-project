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
filename = r'C:\\Users\\215-01\\Desktop\\빅콘\\gen처리.py'
dest = r'C:\\Users\\215-01\\Desktop\\빅콘\\gen처리.ipynb'
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
#train셋의 user id로 탄생년도와 성별 채우기
# 두 변수는 개인의 고유적인 특징, 불변이기 때문에 채워줌
train_gen['birth_year'] = train_gen['birth_year'].fillna(train_gen.groupby('user_id')['birth_year'].transform('mean'))
train_gen['gender'] = train_gen['gender'].fillna(train_gen.groupby('user_id')['gender'].transform('mean'))
#%%
test_gen['birth_year'] = test_gen['birth_year'].fillna(train_gen.groupby('user_id')['birth_year'].transform('mean'))
test_gen['gender'] = test_gen['gender'].fillna(train_gen.groupby('user_id')['gender'].transform('mean'))
#%%
# 나눠진 데이터 셋들의 정보에는, 개인 회생을 신청한 사람과, 신청하지 않은 사람의 정보가 담겨있음.
# 정보가 다 담겨 있기 때문에, 굳이 결측을 채워주지 않고, 변수를 제거하고 사용
# 개인회생 여부 포함하면서 나눠줌, gen, unu 셋에는 개인회생을 신청한 애들과, 신청안한 애들
# 나눠짐, 변수 제거해도 그 속성은 남아있어서 변수 제거
# unu는 개인회생 완납, 미납 차이있는지 살펴봐야해서 완납은 살려 둠
train_gen.drop(['personal_rehabilitation_yn','personal_rehabilitation_complete_yn'],axis=1,inplace =True)
test_gen.drop(['personal_rehabilitation_yn','personal_rehabilitation_complete_yn'],axis=1,inplace =True)
#%%
'''
기대출수가 존재하지만, 기대출금액이 0인 경우: 금융사에서 금액을 제공하지 않은 경우
결측치 채우기 위해 확인 과정
''' 
print(train_gen[(train_gen['existing_loan_amt']==0)&(train_gen['existing_loan_cnt']>=1)])
print('\t')
print(test_gen[(test_gen['existing_loan_amt']==0)&(test_gen['existing_loan_cnt']>=1)])
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
#%%
# 기대출 수가 있지만 금융사에서 제공해주지 않아 amt변수의 값이 0인 것들은 기대출수의 따른 평균 값으로 대체
train_gen.loc[(train_gen['existing_loan_cnt'] == 1) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 35353221
train_gen.loc[(train_gen['existing_loan_cnt'] == 2) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 57461998
train_gen.loc[(train_gen['existing_loan_cnt'] == 3) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 75134303
train_gen.loc[(train_gen['existing_loan_cnt'] == 4) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 90688457
train_gen.loc[(train_gen['existing_loan_cnt'] == 5) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 105104832
train_gen.loc[(train_gen['existing_loan_cnt'] == 6) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 113855294
train_gen.loc[(train_gen['existing_loan_cnt'] == 13) & (train_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 151771285

#%%
# cnt 변수의 nan 값은 기대출 수가 없는 사람으로 판단하고 cat, amt 변수들의 nan값들을 다 0으로 채워줌
test_gen.loc[test_gen['existing_loan_cnt'] != test_gen['existing_loan_cnt'], 'existing_loan_cnt'] = 0
test_gen.loc[test_gen['existing_loan_amt'] != test_gen['existing_loan_amt'], 'existing_loan_amt'] = 0
#%%
# 기대출 수가 있지만 금융사에서 제공해주지 않아 amt변수의 값이 0인 것들은 기대출수의 따른 평균 값으로 대체
test_gen.loc[(test_gen['existing_loan_cnt'] == 1) & (test_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 35353221
test_gen.loc[(test_gen['existing_loan_cnt'] == 2) & (test_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 57461998
test_gen.loc[(test_gen['existing_loan_cnt'] == 3) & (test_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 75134303
test_gen.loc[(test_gen['existing_loan_cnt'] == 4) & (test_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 90688457
test_gen.loc[(test_gen['existing_loan_cnt'] == 5) & (test_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 105104832
test_gen.loc[(test_gen['existing_loan_cnt'] == 6) & (test_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 113855294
test_gen.loc[(test_gen['existing_loan_cnt'] == 13) & (test_gen['existing_loan_amt'] == 0),'existing_loan_amt'] = 151771285

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
# 근속으로 만들기
train_gen['근속년도'] = 2022 - train_gen['입사_년도']  
train_gen['근속개월'] = train_gen['근속년도']*12 + train_gen['입사_월']  
#%%
test_gen['근속년도'] = 2022 - test_gen['입사_년도']  
test_gen['근속개월'] = test_gen['근속년도']*12 + test_gen['입사_월']
#%%
train_gen['age'] = 2022 - train_gen['birth_year']  
test_gen['age'] = 2022 - test_gen['birth_year']  
#%%
train_gen.to_csv('train_gen.csv',encoding = 'cp949')
test_gen.to_csv('test_gen.csv',encoding = 'cp949')
#%%
train_gen = pd.read_csv('train_gen.csv',encoding = 'cp949')
test_gen = pd.read_csv('test_gen.csv',encoding = 'cp949')
#%%
print(train_gen.isnull().sum())
print(test_gen.isnull().sum())
#%%
print(train_gen.dtypes)
#%%
# 모델링 과정, 변수 버리기, 수치형들만 놔두기
# 라벨 인코딩을 해야하므로 결국 범주형 변수들도 수치형으로 사용해야 함
# 변수 내 1,2,3,,,,200까지 있기 때문에 스케일링을 통해 그 영향을 줄임
# type 변수는 문자형 그대로, 더미변수로 생성
# 변수변환에 사용해준 변수도 버림
train_gen.drop(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'application_id','birth_year',
       'loanapply_insert_time','company_enter_month','insert_time','입사_년도','입사_월',
       '근속년도','month','user_id'],axis=1,inplace = True)
#%%
test_gen.drop(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'application_id','birth_year',
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
train_gen_age.to_csv('train_gen_age.csv')
train_gen_credit.to_csv('train_gen_credit.csv')
train_gen_drop_na.to_csv('train_gen_drop_na.csv')
train_gen_enter.to_csv('train_gen_enter.csv')
train_gen_loan.to_csv('train_gen_loan.csv')
#%%
train_gen_age.to_csv('train_gen_age_cp.csv',encoding = 'cp949')
train_gen_credit.to_csv('train_gen_credit_cp.csv',encoding = 'cp949')
train_gen_drop_na.to_csv('train_gen_drop_na_cp.csv',encoding = 'cp949')
train_gen_enter.to_csv('train_gen_enter_cp.csv',encoding = 'cp949')
train_gen_loan.to_csv('train_gen_loan_cp.csv',encoding = 'cp949')

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
test_gen_age.to_csv('test_gen_age.csv')
test_gen_credit.to_csv('test_gen_credit.csv')
test_gen_drop_na.to_csv('test_gen_drop_na.csv')
test_gen_enter.to_csv('test_gen_enter.csv')
test_gen_loan.to_csv('test_gen_loan.csv')
#%%
test_gen_age.to_csv('test_gen_age_cp.csv',encoding = 'cp949')
test_gen_credit.to_csv('test_gen_credit_cp.csv',encoding = 'cp949')
test_gen_drop_na.to_csv('test_gen_drop_na_cp.csv',encoding = 'cp949')
test_gen_enter.to_csv('test_gen_enter_cp.csv',encoding = 'cp949')
test_gen_loan.to_csv('test_gen_loan_cp.csv',encoding = 'cp949')

#%%
print(train_gen.columns)
#%%
train_gen_drop_na = pd.read_csv('train_gen_drop_na.csv')
#%%
test_gen_drop_na = pd.read_csv('test_gen_drop_na.csv')

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
# test_gen 스케일링을 위한 수치형, 범주형 나누기
test_gen_num = test_gen_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_gen_ob = test_gen_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
#%%
# 이상치가 존재하므로 수치형 변수 gen 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
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

test_gen_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
# 나눠준 데이터 합치기 concat
train_gen_sca = pd.concat([train_gen_ob,train_gen_scaled],axis=1)
test_gen_sca = pd.concat([test_gen_ob,test_gen_scaled],axis=1)
#%%
print(train_gen_sca.head())

#%%
train_gen_sca.to_csv('train_gen_sca.csv')

#%%
test_gen_sca.to_csv('test_gen_sca.csv')
#%%
test_gen_sca.to_csv('test_gen_sca_cp.csv',encoding = 'cp949')
#%%
test_gen_sca.to_csv('test_gen_sca_cp.csv',encoding = 'cp949')
#%%
train_gen_sca = pd.read_csv('train_gen_sca.csv')
#%%
dum_t_g = pd.get_dummies(train_gen_sca)
#%%
train_gen_sca = train_gen_sca.astype({'is_applied':'int64','gender':'int64',
                                      'bank_id':'int64','product_id':'int64',
                                      'credit_score':'int64','desired_amount':'int64',
                                      'existing_loan_cnt':'int64','existing_loan_amt':'int64',
                                      '근속개월':'int64','age':'int64',
                                      'yearly_income':'int64','loan_limit':'int64'})
#%%
test_gen_sca = test_gen_sca.astype({'is_applied':'int64','gender':'int64',
                                      'bank_id':'int64','product_id':'int64',
                                      'credit_score':'int64','desired_amount':'int64',
                                      'existing_loan_cnt':'int64','existing_loan_amt':'int64',
                                      '근속개월':'int64','age':'int64',
                                      'yearly_income':'int64','loan_limit':'int64'})


#%%
# train, validation 나누기
train_gen_x = train_gen.drop('is_applied',axis=1)
train_gen_y = train_gen['is_applied']

x_train, x_val, y_train, y_val = train_test_split(train_gen_x,train_gen_y, test_size=0.4, random_state=777)

#%%
#수치형 변수만 남기기
#tr_gen_num = train_gen.copy().drop(['application_id', 'loanapply_insert_time', 'bank_id', 'product_id',
#       'user_id', 'birth_year','gender', 'insert_time','income_type',
#       'company_enter_month', 'employment_type', 'houseown_type',
#       'purpose', 'existing_loan_cnt','existing_loan_amt','month','Unnamed: 0'],axis=1)

tr_gen_num = train_gen.copy()[['loan_limit','loan_rate','birth_year','credit_score','yearly_income',
                               'month','desired_amount',
                               'existing_loan_cnt','existing_loan_amt']]

#다중공선성
tr_gen_num = tr_gen_num.dropna(axis=0)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(tr_gen_num.values, i) for i in range(tr_gen_num.shape[1])]
vif["features"] = tr_gen_num.columns
print(vif)
# 다중공선성 없음
