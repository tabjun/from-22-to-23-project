import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.stats.multicomp import MultiComparison
import scipy.stats
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm
import random
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
# 고객유형별 대출여부 파악을 위한 user + is_applied
''' user에 대출 여부만 붙여서 고객 유형별 대출 보려고 했는데 
    합쳐지는 과정에서 6월에는 결측이어야 할 대출여부 값에 0과1이 생김
    그래서 left조인으로 정확하게 파악해봐야 할 필요가 있음
#%%
# user_spec에 is_applied 붙여주기, 고객 유형별로 대출 신청하는지 파악하기 위해
# user is_applied변수를 만드는데 loan_application_id를 기준으로 is_applied
user['is_applied'] = loan.groupby(['application_id'])['is_applied'].transform('sum')
#user['is_applied_1'] = loan.groupby(['application_id'])['is_applied'].apply(lambda x: 1 if x >= 1 else 0 if x == 0 else x)
#%%
# 제대로 들어갔는지 확인
print(user[user['is_applied']>=1])
print(user.info())
print(user.isnull().sum())
#%%
# 예 아니오 값만 하기 위해 0이면 0 0보다 크면 1로 변경
user['대출신청여부'] = user['is_applied'].apply(lambda x: 1 if x >= 1 else 0 if x == 0 else x)


#%%
# df만들기
user.drop('is_applied',axis=1,inplace = True)
#%%
df = user.copy()
# 제대로 담겼는지 확인
print(df.info())
print(df.isnull().sum())
#%%
# insert_time 데이트타입으로 변경후 월만 있는 파생변수 생성
df['insert_time'] = pd.to_datetime(df['insert_time'])
df['insert_month'] = df['insert_time'].dt.month
#%%
print(df[df['insert_month']==6]['대출신청여부'].value_counts())
a = df[df['insert_month']==6]
#%%
# user에 넣은 is_applied에 6월달에 값이 존재, 0은 transform으로 연산하면서 결측치가 0으로 계산된다고 쳐도,
# 1은 왜있는지. 이게 application_id로 합치다 보니 날짜별로 계산 안해서 그런듯
# 근데 애초에 application_id가 날짜별로 개인이 조회할 때마다 달라지는데 이것도 말이 안됨.
# loan에 application_id적용시켜봤을 때 다 결측임을 확인 
a = df[df['insert_month']==6]['application_id'].tolist()
app = loan[loan['application_id'].isin(a)]
app.sort_values(['application_id'],inplace =True)

b = df[df['insert_month']!=6]['application_id'].tolist()
ab = loan[loan['application_id'].isin(b)]
ab.sort_values(['application_id'],inplace =True)
print(ab.isnull().sum())
#%%
ab_null = ab[ab['is_applied'].isnull()==True]
#%%
'''
#%%
# left join으로 진행
# user 와 loan 데이터 셋의 application_id 정렬 이후 
sort_loan = loan.sort_values('application_id')
sort_log = log.sort_values('user_id')
sort_user = user.sort_values('application_id')
#%%
print(len(user['user_id'].unique()))
#%%
'''
outer로 진행할 경우 is_applied에 생기는 결측치 채우기가 애매함.
outer = pd.merge(sort_loan, sort_user, left_on='application_id', 
                 right_on='application_id', how='outer')
#%%
outer.drop(['bank_id','product_id','loan_rate','loan_limit'],axis=1,inplace =True)
'''
#%%
left = pd.merge(sort_loan, sort_user, left_on='application_id', 
                 right_on='application_id', how='left')
left.drop(['bank_id','product_id','loan_rate','loan_limit'],axis=1,inplace =True)
#%%
''' outer의 개인회생과 left의 개인회생 살펴보기, 개인회생이 1일 때는 1만개, left조인할 때 outer와,
application_id 차이가 30만개, 이중 포함 돼있으면, 개인회생이라는 유저 정보를 잃어 버릴수도 있음
만약에 확인해보고 차이가 없으면, left사용 
비교 결과 별 차이 없음. 
'''
app = left[left['user_id'].isin(lid)]
print(app['personal_rehabilitation_yn'].value_counts())
''' 
결과는 
0.00    7597323   41226/(9579323+41226)
1.00      41226   0.004285202434913018
'''
ab = outer[outer['user_id'].isin(oid)]
print(ab['personal_rehabilitation_yn'].value_counts())
'''
결과
0.00    7845237  49077/(7845237+49077)
1.00      49077  0.006216752969289035
'''
#%%
print(user['personal_rehabilitation_yn'].value_counts())
'''
0.00    794046    12709/(12709+794046)
1.00     12709    0.015753233633507075
'''
#%%
'''
left join으로 안하고, outer로 하면 loan에 없는 user_spec에 존재하는 application_id가 존재하면
is_applied가 결측치가 생기고, 그걸 임의로 채우는 것은 위험한 정보
'''
#%%
sns.countplot(left['personal_rehabilitation_yn'])
plt.title('개인회생 여부')
plt.show()
#%%
sns.countplot(left['is_applied'])
plt.title('대출 여부')
plt.show()
#%%
print(left['personal_rehabilitation_yn'].value_counts())
print(left['is_applied'].value_counts())



#%%
''' 결측치 처리'''
'''
fillna(df.groupby('Gender')['Fruit'].transform(lambda x: x.value_counts().idxmax()), inplace=True)
최댓값으로 결측치 채우기
'''
print(left.isnull().sum())
left_copy = left.copy()
#%%
#datetime변환을 이용하여 6월 데이터 개수 확인
left_copy['loanapply_insert_time'] = pd.to_datetime(left_copy['loanapply_insert_time'])
#%%
left_copy['month'] = pd.DatetimeIndex(left_copy['loanapply_insert_time']).month
#%%
# user_id를 비롯한 113개 행 모두 같은 결측에, appli로 채우려고 해도 다 하나씩만 존재해서 불가능
# 제거
# 6월 데이터 있는지 확인, 존재하지 않음 제거 해도 됨
print((left_copy[left_copy['user_id'].isnull()]['month'].unique()))
#%%
print(len(left_copy[left_copy['user_id'].isnull()]['application_id'].unique()))
left_copy.dropna(subset = ['user_id'],axis=0,inplace = True)
print(left_copy.isnull().sum())
#%%
# yearly_income 제거
# 같은 날 조회한 값에서 0, fillna(0)
a = left_copy[left_copy['yearly_income'].isnull()]
y_a = left_copy[left_copy['user_id']==670502]
left_copy['yearly_income'].fillna(0,inplace = True)
print(left_copy.isnull().sum())

#%%
# 
train = left_copy[left_copy['month']!=6]
test = left_copy[left_copy['month']==6]
#%%

left_copy = left_copy.replace('LIVING','생활비')
left_copy = left_copy.replace('SWITCHLOAN', '대환대출')
left_copy = left_copy.replace('BUSINESS', '사업자금')
left_copy = left_copy.replace('ETC', '기타')
left_copy = left_copy.replace('HOUSEDEPOSIT', '전월세보증금')
left_copy = left_copy.replace('BUYHOUSE', '주택구입')
left_copy = left_copy.replace('INVEST', '투자')
left_copy = left_copy.replace('BUYCAR', '자동차구입')
#%%
#기대출금액 0이면서 기대출수 1 , 6월 이상인 것, 기대출금액이 0인 것 2만개 존
print(left_copy[(left_copy['existing_loan_amt']==0)&(left_copy['month']==6)])
#%%
print(left_copy[(left_copy['birth_year'].isnull())&(left_copy['month']==6)])
#%%
pdf1 = pd.pivot_table(left_copy,                # 피벗할 데이터프레임
                     index = 'user_id',    # 행 위치에 들어갈 열
                     columns = 'application_id', # 열 위치에 들어갈 데이터: application_id
                     values = 'birth_year',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
pdf1
#%%
pdf1['float'] = pdf1%1
print(pdf1[pdf1['float']!=0])
#%%
# birth_year groupby를 이용한 결측치 처리
left_copy['birth_year'].fillna(pdf1.groupby(['user_id'])['birth_year'].transform('mean'),inplace = True)
print(left_copy.isnull().sum())
#%%
# birth_year groupby를 이용한 결측치 처리, 효과 없음
left_copy['birth_year'].fillna(left_copy.groupby(['user_id'])['birth_year'].transform('mean'),inplace = True)
print(left_copy.isnull().sum())
#%%
# birth_year groupby를 이용한 결측치 처리
left_copy['birth_year'].fillna(user.groupby(['user_id'])['birth_year'].transform('mean'),inplace = True)
print(left_copy.isnull().sum())
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
'''
print(f"개인회생여부 0:{len(df_copy[df_copy['personal_rehabilitation_yn']==0])}")
print(f"개인회생여부 1 :{len(df_copy[df_copy['personal_rehabilitation_yn']==1])}")
print(f"개인회생여부 결측 :{len(df_copy[df_copy['personal_rehabilitation_yn'].isnull()==True])}")
print(df_copy['personal_rehabilitation_complete_yn'].value_counts())
# 개인 회생여부 1인애들 제거 , 일단 일반적인 경우에 대출을 할 경우를 판단하기 위해
# 개인 회생절차를 받은 애들은 제거해주고 나머지 결측치는 0으로 채움
# 잠시 보류 
df_copy = df_copy[df_copy['personal_rehabilitation_yn']!=1].copy() '''
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
# id에 따른 탄생년도 최대값 최솟값 비교로 다른 값 존재여부 확인
pdf_max = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'user_id',    # 행 위치에 들어갈 열
                     values = 'birth_year',     # 데이터로 사용할 열
                     aggfunc = 'max')   # 데이터 집계함수

#%%
pdf_min = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'user_id',    # 행 위치에 들어갈 열
                     values = 'birth_year',     # 데이터로 사용할 열
                     aggfunc = 'min')   # 데이터 집계함수

#%%
pdf_min['차이'] = pdf_max.birth_year - pdf_min.birth_year
print(pdf_min.차이.value_counts())
print(f'birth_year차이 :0 , 개수: {len(pdf_min.차이)}')
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
