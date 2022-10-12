
# In[372]:
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


# In[373]:


pd.options.display.float_format = '{:.2f}'.format

#%%
# 경로지정
os.chdir('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908')
os.getcwd()
# In[374]:


loan = pd.read_csv('loan_result.csv')
log = pd.read_csv('log_data.csv')
user = pd.read_csv('user_spec.csv',encoding=('UTF-8'))


# In[375]:


print(loan.info()) # 데이터가 너무 커서 dtype이 안나옴
print(loan.dtypes)
print(loan.isnull().sum())
print(loan.describe(include='all'))


# In[376]:


print(log.info())
print(log.dtypes)
print(log.isnull().sum())
print(log.describe(include='all'))


# In[377]:


print(user.info())
print(user.dtypes)
print(user.isnull().sum())
print(user.describe(include='all'))


# In[378]:


#log_data의 어플 버전, 안드로이드 ios구분, 값이 다르다. 
# 태경 안드로이드 폰으로 버전 보여주고, 소문자 값이 휴대폰인지 확인
print(log['mp_os'].value_counts())
print(log['mp_app_version'].value_counts())


# In[379]:


# left join으로 진행
# user 와 loan 데이터 셋의 application_id 정렬 이후 
sort_loan = loan.sort_values('application_id')
sort_log = log.sort_values('user_id')
sort_user = user.sort_values('application_id')

left = pd.merge(sort_loan, sort_user, left_on='application_id', 
                 right_on='application_id', how='left')
print(left.isnull().sum())


# In[380]:


left_copy = left.copy()


# In[381]:


#datetime변환을 이용하여 6월 데이터 개수 확인
left_copy['loanapply_insert_time'] = pd.to_datetime(left_copy['loanapply_insert_time'])


# In[382]:


# datetime변수에서 month추출
left_copy['month'] = pd.DatetimeIndex(left_copy['loanapply_insert_time']).month


# In[383]:


# purpose 값 바꿔주기
left_copy = left_copy.replace('LIVING','생활비')
left_copy = left_copy.replace('SWITCHLOAN', '대환대출')
left_copy = left_copy.replace('BUSINESS', '사업자금')
left_copy = left_copy.replace('ETC', '기타')
left_copy = left_copy.replace('HOUSEDEPOSIT', '전월세보증금')
left_copy = left_copy.replace('BUYHOUSE', '주택구입')
left_copy = left_copy.replace('INVEST', '투자')
left_copy = left_copy.replace('BUYCAR', '자동차구입')


# In[ ]:


left_copy.to_csv('left.csv')


# In[384]:


left = pd.read_csv('left.csv')
left_copy = left.copy()


# In[385]:


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

# In[]
train_unu.to_csv('train_unu.csv',encoding = 'cp949')
test_unu.to_csv('test_unu.csv',encoding = 'cp949')
# In[]
train_unu = pd.read_csv('train_unu.csv',encoding = 'cp949')
test_unu = pd.read_csv('test_unu.csv',encoding = 'cp949')
# In[386]:


# gen의 개인 회생, 납입 완료 값 확인 0 또는 결측만 있음, 잘 담김
print('일반 훈련셋')
print(train_gen[['personal_rehabilitation_yn','personal_rehabilitation_complete_yn']].value_counts())
print('일반 테스트 셋')
print(test_gen[['personal_rehabilitation_yn','personal_rehabilitation_complete_yn']].value_counts())
print('특별 훈련셋')
print(train_unu[['personal_rehabilitation_yn','personal_rehabilitation_complete_yn']].value_counts())
print('특별 테스트')
print(test_unu[['personal_rehabilitation_yn','personal_rehabilitation_complete_yn']].value_counts())


# In[387]:


print(train_unu.personal_rehabilitation_complete_yn.value_counts())
print(test_unu.personal_rehabilitation_complete_yn.value_counts())


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

# ### 스케일링

# In[476]:


# 이상치가 존재하므로 수치형 변수 gen 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
train_unu_scaled = rbs.fit_transform(train_unu_num) #fit시킨 데이터 적용
test_unu_scaled = rbs.transform(test_unu_num) #fit시킨 데이터 적용

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
# py , ipynb 변환
import subprocess
filename = r'C:\\Users\\215-01\\Desktop\\빅콘\\빅콘unu.py'
dest = r'C:\\Users\\215-01\\Desktop\\빅콘\\unu처리.ipynb'
subprocess.run(['ipynb-py-convert', filename, dest])
#%%
print(train_unu_age.isnull().sum())
print(test_unu_age.isnull().sum())
#%%
ax.set_ylim([y.min()-0.05, y.max()+0.05])
try:
    ax.set_ylim([y.min()-0.05, y.max()+0.05])
except ValueError:  #raised if `y` is empty.
    pass
#%%
for i in train_unu_age.columns:
    model = ols('gender ~ train_unu_age[i]', train_unu_age).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*2)
    print(model.summary())
    print('\n')