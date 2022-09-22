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
# user_spec에 is_applied 붙여주기, 고객 유형별로 대출 신청하는지 파악하기 위해
# user is_applied변수를 만드는데 loan_application_id를 기준으로 is_applied
user['is_applied'] = loan.groupby(['application_id'])['is_applied'].transform('sum')
#%%
# 제대로 들어갔는지 확인
print(user[user['is_applied']>=1])
print(user.info())
#%%
# 예 아니오 값만 하기 위해 0이면 0 0보다 크면 1로 변경
user['대출신청여부'] = user['is_applied'].apply(lambda x: 1 if x >= 1 else 0 if x == 0 else x)
#%%
# 85개 공통 결측 행 제거, 0.006%의 비율을 차지하고 있는 극 소량의 데이터이며, 조회과정에서
# 소득정보 입력에 대해 무관이라고 선택을 할 수도 있음.
df_copy = df.copy()
print(df_copy.isnull().sum())
df_copy.dropna(subset=['employment_type'],inplace = True)
print('\t')
print(df_copy.isnull().sum())
#%%
# yearly_income을 채우는 과정
yearly_none = df_copy[df_copy['yearly_income'].isnull()==True]
y1 = df_copy[df_copy['user_id'] == 702899 ]
y2 = df_copy[df_copy['user_id'] == 861363 ]
y3 = df_copy[df_copy['user_id'] == 329226 ]
y4 = df_copy[df_copy['user_id'] == 670502 ]
y5 = df_copy[df_copy['user_id'] == 771592 ]
#%%
df_copy['yearly_income'].fillna(0,inplace = True)
#%%
# 얘가 매번 연봉도 다르고, 대출 조건 조회 기간마다 입사년월이 달라짐, 이상한 놈
# 같은 날 연봉입력한거에는 1천만원 그니까 이걸로 채워넣음
df_copy.loc[233316, 'yearly_income'] = 10000000.0
#%%
# id : 861363 행 제거 및 결측치 0 채우기
df_copy = df_copy[df_copy['user_id']!=861363]
#%%
print(df_copy.isnull().sum())
#%%
print(f"개인회생여부 0:{len(df_copy[df_copy['personal_rehabilitation_yn']==0])}")
print(f"개인회생여부 1 :{len(df_copy[df_copy['personal_rehabilitation_yn']==1])}")
print(f"개인회생여부 결측 :{len(df_copy[df_copy['personal_rehabilitation_yn'].isnull()==True])}")
print(df_copy['personal_rehabilitation_complete_yn'].value_counts())
# 개인 회생여부 1인애들 제거 , 일단 일반적인 경우에 대출을 할 경우를 판단하기 위해
# 개인 회생절차를 받은 애들은 제거해주고 나머지 결측치는 0으로 채움
# 잠시 보류
#%%
# df_copy = df_copy[df_copy['personal_rehabilitation_yn']!=1].copy()
#%%
# 입사년월이 탄생년도보다 빠른애들 제거해주기 위해 입사년월 분리
# 입사년월과 탄생년도 비교를 위해 입사연도와 입사월으로 분리해주기
df_over = df_copy[df_copy['company_enter_month']>=202207].sort_values(['company_enter_month']) 
df_under = df_copy[df_copy['company_enter_month']<202207].sort_values(['company_enter_month'])
df_null = df_copy[df_copy['company_enter_month'].isnull()==True] 
#%%
# 개수 확인, 제대로 나뉘었는지 확인
print(df_copy[df_copy['company_enter_month']>=202207])
print(df_copy[df_copy['company_enter_month']<202207])
print(df_copy[df_copy['company_enter_month'].isnull()==True])
#%%
# df_over 데이터 프레임을 살펴보았을 때 202206을 벗어난 값이 존재함.
print(df_copy[df_copy['company_enter_month']==202211])
print(df_copy[df_copy['user_id']==563134]['company_enter_month'])
#%%
# 다른 값을 확인해보았을 때 202201에 입사했다.
df_copy['company_enter_month'] = df_copy['company_enter_month'].apply(lambda x: 202201 if x == 202211 else x)
#%%
# 입사년월이 6자리인 애들
df_under['입사_년도'] = df_under['company_enter_month']//100
df_under['입사_월'] = df_under['company_enter_month']%100
#%%
# 확인했을 때 잘 분리됨
print(df_under['입사_년도']) 
print(df_under['입사_월'])
#%%
# 입사년월이 8자리인 애들
df_over['입사_년도'] = df_over['company_enter_month']//10000
df_over['입사_월'] = (df_over['company_enter_month']//100)%100
#%%
# 확인했을 때 잘 분리됨
print(df_over['입사_년도']) 
print(df_over['입사_월'])
#%%
# 각각 입사년월 나눠준 데이터 셋 합치기
df_1 = pd.concat([df_under, df_over], axis = 0)
df_merge = pd.concat([df_1, df_null], axis = 0)
#%%
# 합친거 잘 합쳐졌나 확인
print(df_merge[df_merge['company_enter_month']>=202207])
df_merge[df_merge['company_enter_month']<202207]
print(df_merge[df_merge['company_enter_month'].isnull()==True])
#%%
pdf1 = pd.pivot_table(df_merge,                # 피벗할 데이터프레임
                     index = 'user_id',    # 행 위치에 들어갈 열
                     values = 'birth_year',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
pdf1
#%%
pdf1['float'] = pdf1%1
print(pdf1[pdf1['float']!=0])
# user_id별로 확인을 해보았을 때 , 소숫점자리가 존재하는 값이 없음
# 즉 탄생년도를 대출 조회시 다르게 입력한 사람이 없다. 
#%%
# 입사년월이 결측치일때 id 개수 확인
print(len(df_merge[df_merge['company_enter_month'].isnull()==True]['user_id'].unique()))
#%%
# birth_year groupby를 이용한 결측치 처리
df_merge['birth_year'].fillna(df_merge.groupby(['user_id'])['birth_year'].transform('mean'),inplace = True)
print(df_merge.isnull().sum())
#%%
# birth_year결측치가 다 처리되지 않음, 왜 그런지 파악하기 위해 처리되지 않은 결측치들의
# user_id를 살펴봄
a = df_merge[df_merge['birth_year'].isnull()==True]
print(a['user_id'].head())
print(a['user_id'].tail())
b = df_copy[df_copy['birth_year'].isnull()==True]
print(b['user_id'].head())
print(b['user_id'].tail())
#%%
print(df_copy[df_copy['user_id']==284238]['birth_year'])
print(f'744785 :{df_copy[df_copy["user_id"]==744785]["birth_year"]}')
print(df_copy[df_copy['user_id']==77317]['birth_year'])
#%%
# 해결을 위해 id의 탄생년도 결측치 행을 다 제거한 데이터프레임의 groupby.mean으로 결측치 채우기
no_birth = df_merge[df_merge['birth_year'].isnull()==False].copy()
df_merge['birth_year'].fillna(no_birth.groupby(['user_id'])['birth_year'].transform('mean'),inplace = True)
print(df_merge.isnull().sum())
# 달라지지 않음
#%%
# 아까 만들어놓은 피벗테이블을 이용해 결측치 처리
df_merge['birth_year'].fillna(pdf1.groupby(['user_id'])['birth_year'].transform('mean'),inplace = True)
print(df_merge.isnull().sum())
#%%
#gender 확인
pdf2 = pd.pivot_table(df_merge,                # 피벗할 데이터프레임
                     index = 'user_id',    # 행 위치에 들어갈 열
                     values = 'gender',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
pdf2
#%%
# 성별이 0과 1말고는 없음을 확인, 즉 같은 id내 다른 값 존재하지 않음.
print(pdf2[(pdf2['gender']!=0)&(pdf2['gender']!=1)])
#%%
# 아까 만들어놓은 피벗테이블을 이용해 결측치 처리
df_merge['gender'].fillna(pdf2.groupby(['user_id'])['gender'].transform('mean'),inplace = True)
print(df_merge.isnull().sum())
#%%
d_n = df_merge[df_merge['gender'].isnull()==True]
# 확인 결과 모두 같은 행에 존재, 결측치 꾸역꾸역 채워봤자 정확하지 않음
#%%
# 9166개 행 제거 예정
# birth_year id별로 좀 더 확인이 필요하다 말했지만
df_merge.dropna(subset=['gender'],inplace = True)
print(df_merge.isnull().sum())
