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
print(user['대출신청여부'].value_counts())
#%%
df = pd.read_csv('고객유형.csv')
print(df.head())
#%%
df = df.drop('Unnamed: 0',axis=1)
#%%
print(df[df['대출신청여부']==1])
#%%
print(df.isnull().sum())
#%%
a = df[df['purpose'].isnull()==True]
#%%
a.to_csv('nu_Df.csv')
#%%
a = df.isnull().sum()
a.to_csv('nu.csv')
#%%
a = a.sort_values('user_id')
a_des = a.describe()
#%%
df['신용등급'] = df['credit_score'].apply(lambda x: 1 if 942 <=  x <= 1000 
                                                else 2 if 891<= x <= 941 else 3 if 832<=x<=890 else 4 if 768 <= x <= 831
                                               else 5 if 698 <= x <= 767 else 6 if 630 <= x <= 697 else 7 if 530 <= x <= 629
                                               else 8 if 454 <= x <= 529 else 9 if 335 <= x <= 453 else 10 if 0 <= x <= 334 else x)
#%%
no_user = log[(log['user_id']<4325)&(log['event'] != 'SignUp')]
#%%
print(df['employment_type'].value_counts())
print(df['income_type'].value_counts())
#%%
# 85개의 결측치를 채워주기 위해, 결측치가 있다는 것은 없다는 정보일 수도
# 그냥 바로 무직이라기엔 신용점수가 높으며, 대출승인을 해준 경험이 있음
# 값이 존재하는 것 중에 employment_type : 기타, income_type: 기타소득(무직,주부,학생)인
# 애들의 신용점수도 확인하여 비슷한 형태를 띄고 있으면 결측값 무직으로 때려줌
a2= df[df['employment_type']=='기타']
a3= df[df['employment_type']=='정규직']
a4= df[df['employment_type']=='계약직']
a5= df[df['employment_type']=='일용직']
print(a2['income_type'].value_counts())
#%%
계약_근로 = df[(df['employment_type']=='계약직')&(df['income_type']=='EARNEDINCOME')]
계약_사업 = df[(df['employment_type']=='계약직')&(df['income_type']=='PRIVATEBUSINESS')]
계약_기타 = df[(df['employment_type']=='계약직')&(df['income_type']=='OTHERINCOME')]
계약_근로2 = df[(df['employment_type']=='계약직')&(df['income_type']=='EARNEDINCOME2')]
계약_프리 = df[(df['employment_type']=='계약직')&(df['income_type']=='FREELANCER')]
계약_전문 = df[(df['employment_type']=='계약직')&(df['income_type']=='PRACTITIONER')]

일용_근로 = df[(df['employment_type']=='일용직')&(df['income_type']=='EARNEDINCOME')]
일용_사업 = df[(df['employment_type']=='일용직')&(df['income_type']=='PRIVATEBUSINESS')]
일용_기타 = df[(df['employment_type']=='일용직')&(df['income_type']=='OTHERINCOME')]
일용_근로2 = df[(df['employment_type']=='일용직')&(df['income_type']=='EARNEDINCOME2')]
일용_프리 = df[(df['employment_type']=='일용직')&(df['income_type']=='FREELANCER')]
일용_전문 = df[(df['employment_type']=='일용직')&(df['income_type']=='PRACTITIONER')]
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(일용_근로['credit_score'],label='일용_근로 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 일용_근로 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(일용_사업['credit_score'],label='일용_사업 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 일용_사업 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(일용_기타['credit_score'],label='일용_기타 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 일용_기타 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(일용_근로2['credit_score'],label='일용_근로2 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 일용_근로2 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(일용_프리['credit_score'],label='일용_프리 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 일용_프리 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(일용_전문['credit_score'],label='일용_전문 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 일용_전문 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(계약_전문['credit_score'],label='계약_전문 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 계약직 전문 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(계약_프리['credit_score'],label='계약_프리 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 계약직 프리 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(계약_근로2['credit_score'],label='계약_근로2소득 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 계약직 근로2소득 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(계약_기타['credit_score'],label='계약_기타소득 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 계약직 기타소득 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(계약_사업['credit_score'],label='계약_사업소득 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 계약직 개인사업 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(계약_근로['credit_score'],label='계약_근로소득 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 계약직 근로소득 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(a4['credit_score'],label='계약직 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 계약직 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(a3['credit_score'],label='정규직 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 정규직 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(a2['credit_score'],label='기타 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 기타 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(a5['credit_score'],label='일용직 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 일용 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
aabb = df[(df['employment_type']=='기타')&(df['income_type']=='FREELANCER')]
print(aabb['credit_score'].describe())
#%%
#신용점수의 분포가 비슷한지 확인
sns.distplot(aabb['credit_score'],label='일용직 신용점수')
sns.distplot(a['credit_score'],label='결측 신용점')
plt.title('결측치와 일용 신용점수 플랏')
plt.legend(loc='best', title = 'Group')
plt.show()
#%%
#확인해보니 employment_type: 기타, income_type:'OTHERINCOME' 일 때 젤 비슷하다.
df['employment_type'] = df['employment_type'].fillna('기타')
df['income_type'] = df['income_type'].fillna(('OTHERINCOME'))
print(df.isnull().sum())
#%%












