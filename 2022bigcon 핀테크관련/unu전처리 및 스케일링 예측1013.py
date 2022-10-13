#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# credit셋은 따로 모델링 필요없이 drop_na모델이랑 같이

# In[14]:


pd.options.display.float_format = '{:.2f}'.format
#%%
# 경로지정
os.chdir('C:\\Users\\215-01\\Desktop\\빅콘\\2022빅콘테스트_데이터분석리그_데이터분석분야_퓨처스부문_데이터셋_220908')
os.getcwd()
# In[374]:


# In[15]:


train_unu_age = pd.read_csv('train_unu_age_1013.csv')
train_unu_credit = pd.read_csv('train_unu_credit_1013.csv')
train_unu_drop_na = pd.read_csv('train_unu_drop_na_1013.csv')
train_unu_enter = pd.read_csv('train_unu_enter_1013.csv')
train_unu_loan= pd.read_csv('train_unu_loan_1013.csv')


# In[16]:


train_unu_loan.isnull().sum()


# In[18]:


train_unu_loan.drop(['loan_limit','loan_rate'],axis=1,inplace = True)


# In[19]:


for i in train_unu_loan.columns:
    model = ols('credit_score ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')


# In[20]:


#
#credit, personal_rehabilitation_complete_yn
train_unu_loan['credit_score'] = train_unu_loan['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id', 
                                                                                                  'personal_rehabilitation_complete_yn',
                                                                                                 'existing_loan_cnt'])['credit_score'].transform('mean'))


# In[21]:


train_unu_loan.isnull().sum()


# In[47]:


for i in train_unu_loan.columns:
    model = ols('근속개월 ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')


# In[22]:


#
#bank
train_unu_loan['근속개월'] = train_unu_loan['근속개월'].fillna(train_unu_drop_na.groupby(['bank_id'])['근속개월'].transform('mean'))


# In[23]:


train_unu_loan.isnull().sum()


# In[24]:


train_unu_enter.isnull().sum()


# In[25]:


train_unu_enter.drop(['근속개월'],axis=1,inplace = True)


# In[26]:


train_unu_enter.isnull().sum()


# In[27]:


for i in train_unu_enter.columns:
    model = ols('credit_score ~ train_unu_enter[i]', train_unu_enter).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')


# In[28]:


# loan_lim,
#bank,produ,loan_rate,is_applied,gender,employment,purpose,personal_rehabilitation_complete_yn,existing_loan_cnt
train_unu_enter['credit_score'] = train_unu_enter['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id','product_id',
                                                                                                    'loan_rate', 'is_applied',
                                                                                                    'purpose','gender',
                                                                                                    'personal_rehabilitation_complete_yn',
                                                                                                    'existing_loan_cnt'])['credit_score'].transform('mean'))


# In[29]:


train_unu_enter.isnull().sum()


# In[30]:


train_unu_credit.isnull().sum()


# In[31]:


train_unu_credit.drop(['credit_score'],axis=1,inplace = True)


# In[32]:


train_unu_age.isnull().sum()


# In[33]:


train_unu_age.drop(['gender','age'],axis=1,inplace = True)


# In[18]:


for i in train_unu_age.columns:
    model = ols('근속개월 ~ train_unu_age[i]', train_unu_age).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')


# In[34]:


# bank_id,product,income_type(),employment_type,desired_amount,personal_rehabilitation_complete_yn,existing_loan_amt,
# loan_limit,rate,credit_score,houseown_type,purpose,existing_loan_cnt
train_unu_age['근속개월'] = train_unu_age['근속개월'].fillna(train_unu_drop_na.groupby(['loan_limit','loan_rate', 
                                                                                                'credit_score', 
                                                                                                'houseown_type', 'purpose',
                                                                                                'existing_loan_cnt'])['근속개월'].transform('mean'))


# In[35]:


train_unu_age.isnull().sum()


# In[38]:


test_unu_age = pd.read_csv('test_unu_age_1013.csv')
test_unu_credit = pd.read_csv('test_unu_credit_1013.csv')
test_unu_credit = pd.read_csv('test_unu_drop_na_1013.csv')
test_unu_enter = pd.read_csv('test_unu_enter_1013.csv')
test_unu_loan= pd.read_csv('test_unu_loan_1013.csv')


# In[ ]:


train_unu_age['근속개월'] = train_unu_age['근속개월'].fillna(train_unu_drop_na.groupby(['loan_limit','loan_rate', 
                                                                                                'credit_score', 
                                                                                                'houseown_type', 'purpose',
                                                                                                'existing_loan_cnt'])['근속개월'].transform('mean'))


# In[44]:


test_unu_age.drop(['birth_year','gender','age'],axis=1,inplace=True)


# In[45]:


# bank_id,product,income_type(),employment_type,desired_amount,personal_rehabilitation_complete_yn,existing_loan_amt,
# loan_limit,rate,credit_score,houseown_type,purpose,existing_loan_cnt
test_unu_age['근속개월'] = test_unu_age['근속개월'].fillna(train_unu_drop_na.groupby(['loan_limit','loan_rate', 
                                                                                                'credit_score', 
                                                                                                'houseown_type', 'purpose',
                                                                                                'existing_loan_cnt'])['근속개월'].transform('mean'))


# In[47]:


# bank_id,product,income_type(),employment_type,desired_amount,personal_rehabilitation_complete_yn,existing_loan_amt,
# loan_limit,rate,credit_score,houseown_type,purpose,existing_loan_cnt
test_unu_age['credit_score'] = test_unu_age['credit_score'].fillna(train_unu_drop_na.groupby(['existing_loan_cnt'])['근속개월'].transform('mean'))


# In[48]:


test_unu_age.isnull().sum()


# In[49]:


test_unu_credit.isnull().sum()


# In[50]:


test_unu_enter.isnull().sum()


# In[51]:


test_unu_enter.drop(['근속개월'],axis=1,inplace = True)


# In[52]:


# loan_lim,
#bank,produ,loan_rate,is_applied,gender,employment,purpose,personal_rehabilitation_complete_yn,existing_loan_cnt
test_unu_enter['credit_score'] = test_unu_enter['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id','product_id',
                                                                                                    'loan_rate', 'is_applied',
                                                                                                    'purpose','gender','employment_type',
                                                                                                    'personal_rehabilitation_complete_yn',
                                                                                                    'existing_loan_cnt'])['credit_score'].transform('mean'))


# In[53]:


test_unu_enter.isnull().sum()


# In[54]:


test_unu_loan['credit_score'] = test_unu_loan['credit_score'].fillna(train_unu_drop_na.groupby(['bank_id', 
                                                                                                  'personal_rehabilitation_complete_yn',
                                                                                                 'existing_loan_cnt'])['credit_score'].transform('mean'))


# In[55]:


test_unu_loan['근속개월'] = test_unu_loan['근속개월'].fillna(train_unu_drop_na.groupby(['bank_id'])['근속개월'].transform('mean'))


# In[57]:


test_unu_loan.drop(['loan_rate','loan_limit'],axis=1,inplace=True)


# In[58]:


test_unu_loan.isnull().sum()


# In[54]:


for i in train_unu_loan.columns:
    model = ols('gender ~ train_unu_loan[i]', train_unu_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')


# In[60]:


train_unu_loan.gender.value_counts()


# In[59]:


# 최빈값으로 채움
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


# In[62]:


test_unu_loan.to_csv('test_unu_loan_1013_결측처리완.csv')
test_unu_age.to_csv('test_unu_age_1013_결측처리완.csv')
test_unu_enter.to_csv('test_unu_enter_1013_결측처리완.csv')
test_unu_credit.to_csv('test_unu_credit_1013_결측처리완.csv')

train_unu_loan.to_csv('train_unu_loan_1013_결측처리완.csv')
train_unu_age.to_csv('train_unu_age_1013_결측처리완.csv')
train_unu_enter.to_csv('train_unu_enter_1013_결측처리완.csv')
train_unu_credit.to_csv('train_unu_credit_1013_결측처리완.csv')


# In[ ]:





# In[63]:


#%%
# train_unu_age 스케일링을 위한 수치형, 범주형 나누기
train_unu_age_num = train_unu_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']]
#%%
train_unu_age_ob = train_unu_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월'],axis=1)


# In[64]:


#%%
# test_unu_age 스케일링을 위한 수치형, 범주형 나누기
test_unu_age_num = test_unu_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']]
#%%
test_unu_age_ob = test_unu_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월'],axis=1)


# In[66]:


print(train_unu_age.isnull().sum())
print(test_unu_age.isnull().sum())


# In[67]:


#%%
# 이상치가 존재하므로 수치형 변수 unu_age 스케일링
# 결측치가 존재하는 데이터로 정규화해주면, 행 전부 결측치가 됨
# 결측치를 제외한 데이터를 fit, transform으로 적용
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_age) # 결측치 없는 train데이터들로 fit시키고
train_unu_age_scaled = rbs.fit_transform(train_unu_age_num) #fit시킨 데이터 적용
test_unu_age_scaled = rbs.transform(test_unu_age_num) #fit시킨 데이터 적용


# In[68]:


train_unu_age_scaled = pd.DataFrame(data = train_unu_age_scaled )
test_unu_age_scaled = pd.DataFrame(data = test_unu_age_scaled)


# In[70]:


train_unu_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']
#%%
test_unu_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', '근속개월']


# In[72]:


#%%
# train_gen_ob셋은 원래 데이터 셋에서 행들을 제거해준것이기 때문에 인덱스가 일정하지 않음
# train_gen_scaled는 새로 추출해서 한 값이기에 인덱스가 1~800000까지 일정
train_unu_age_scaled.reset_index(drop = False, inplace = True)
train_unu_age_ob.reset_index(drop = False, inplace = True)

test_unu_age_scaled.reset_index(drop = False, inplace = True)
test_unu_age_ob.reset_index(drop = False, inplace = True)


# In[73]:


train_unu_age_sca = pd.concat([train_unu_age_ob,train_unu_age_scaled],axis=1)
test_unu_age_sca = pd.concat([test_unu_age_ob,test_unu_age_scaled],axis=1)


# In[76]:


print(train_unu_age_sca.shape)
train_unu_age.shape


# In[77]:


print(test_unu_age_sca.shape)
test_unu_age.shape


# In[78]:


train_unu_age_sca.to_csv('train_unu_age_sca_1013.csv')
test_unu_age_sca.to_csv('test_unu_age_sca_1013.csv')


# In[80]:


train_unu_enter.columns


# In[81]:


train_unu_enter_num = train_unu_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']]
#%%
train_unu_enter_ob = train_unu_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age'],axis=1)


# In[82]:


test_unu_enter_num = test_unu_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']]
#%%
test_unu_enter_ob = test_unu_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age'],axis=1)


# In[83]:


from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_enter) # 결측치 없는 train데이터들로 fit시키고
train_unu_enter_scaled = rbs.fit_transform(train_unu_enter_num) #fit시킨 데이터 적용
test_unu_enter_scaled = rbs.transform(test_unu_enter_num) #fit시킨 데이터 적용


# In[84]:


train_unu_enter_scaled = pd.DataFrame(data = train_unu_enter_scaled )
test_unu_enter_scaled = pd.DataFrame(data = test_unu_enter_scaled)


# In[85]:


# 변수명 삽입
train_unu_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']
#%%
test_unu_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate',
       'credit_score', 'yearly_income','desired_amount',
       'existing_loan_cnt', 'existing_loan_amt', 'age']


# In[86]:


train_unu_enter_scaled.reset_index(drop = False, inplace = True)
train_unu_enter_ob.reset_index(drop = False, inplace = True)

test_unu_enter_scaled.reset_index(drop = False, inplace = True)
test_unu_enter_ob.reset_index(drop = False, inplace = True)


# In[87]:


train_unu_enter_sca = pd.concat([train_unu_enter_ob,train_unu_enter_scaled],axis=1)
test_unu_enter_sca = pd.concat([test_unu_enter_ob,test_unu_enter_scaled],axis=1)


# In[88]:


print(train_unu_enter_sca.shape)
train_unu_enter.shape


# In[89]:


print(test_unu_enter_sca.shape)
test_unu_enter.shape


# In[90]:


train_unu_enter_sca.to_csv('train_unu_enter_sca_1013.csv')
test_unu_enter_sca.to_csv('test_unu_enter_sca_1013.csv')


# In[91]:


train_unu_loan_num = train_unu_loan.copy()[['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
train_unu_loan_ob = train_unu_loan.copy().drop(['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)


# In[92]:


test_unu_loan_num = test_unu_loan.copy()[['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_unu_loan_ob = test_unu_loan.copy().drop(['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)


# In[93]:


from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_unu_loan) # 결측치 없는 train데이터들로 fit시키고
train_unu_loan_scaled = rbs.fit_transform(train_unu_loan_num) #fit시킨 데이터 적용
test_unu_loan_scaled = rbs.transform(test_unu_loan_num) #fit시킨 데이터 적용


# In[94]:


train_unu_loan_scaled = pd.DataFrame(data = train_unu_loan_scaled )
test_unu_loan_scaled = pd.DataFrame(data = test_unu_loan_scaled)


# In[95]:


train_unu_loan_scaled.columns = ['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_unu_loan_scaled.columns = ['bank_id', 'product_id',
       'credit_score', 'yearly_income', 'desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']


# In[96]:


#%%
# train_gen_ob셋은 원래 데이터 셋에서 행들을 제거해준것이기 때문에 인덱스가 일정하지 않음
# train_gen_scaled는 새로 추출해서 한 값이기에 인덱스가 1~800000까지 일정
train_unu_loan_scaled.reset_index(drop = False, inplace = True)
train_unu_loan_ob.reset_index(drop = False, inplace = True)

test_unu_loan_scaled.reset_index(drop = False, inplace = True)
test_unu_loan_ob.reset_index(drop = False, inplace = True)


# In[97]:


train_unu_loan_sca = pd.concat([train_unu_loan_ob,train_unu_loan_scaled],axis=1)
test_unu_loan_sca = pd.concat([test_unu_loan_ob,test_unu_loan_scaled],axis=1)


# In[98]:


print(train_unu_loan_sca.shape)
train_unu_loan.shape


# In[99]:


print(test_unu_loan_sca.shape)
test_unu_loan.shape


# In[117]:


train_unu_loan_sca.to_csv('train_unu_loan_sca_1013.csv')
test_unu_loan_sca.to_csv('test_unu_loan_sca_1013.csv')


# In[103]:


train_unu_drop_na = pd.read_csv('train_unu_drop_na_1013.csv')


# In[104]:


# train_gen 스케일링을 위한 수치형, 범주형 나누기
train_unu_num = train_unu_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
train_unu_ob = train_cr_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)


# In[133]:


# train_gen 스케일링을 위한 수치형, 범주형 나누기
test_unu_num = test_unu_credit.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_unu_ob = test_unu_credit.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)


# In[134]:


test_unu_drop_na = pd.read_csv('test_unu_drop_na_1013.csv')


# In[135]:


# train_gen 스케일링을 위한 수치형, 범주형 나누기
test_unu_num = test_unu_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
#%%
test_unu_ob = test_unu_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)


# In[139]:


from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
#rbs.fit_transform(no_train_gen) # 결측치 없는 train데이터들로 fit시키고
train_unu_scaled = rbs.fit_transform(train_unu_num) #fit시킨 데이터 적용
test_unu_scaled = rbs.transform(test_unu_num) #fit시킨 데이터 적용


# In[140]:


#%%
train_unu_scaled = pd.DataFrame(data = train_unu_scaled)
test_unu_scaled = pd.DataFrame(data = test_unu_scaled)


# In[141]:


#%%
train_unu_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
#%%
test_unu_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']


# In[112]:


train_unu_scaled.reset_index(drop = False, inplace = True)
train_unu_ob.reset_index(drop = False, inplace = True)

test_unu_cr_scaled.reset_index(drop = False, inplace = True)
test_unu_cr_ob.reset_index(drop = False, inplace = True)


# In[114]:


test_unu_credit_sca = pd.concat([test_unu_cr_ob,test_unu_cr_scaled],axis=1)


# In[115]:


print(test_unu_credit.shape)
test_unu_credit_sca.shape


# In[116]:


test_unu_credit_sca.to_csv('test_unu_credit_sca_1013.csv',index=False)


# In[142]:


train_unu_scaled.reset_index(drop = False, inplace = True)
train_unu_ob.reset_index(drop = False, inplace = True)

test_unu_scaled.reset_index(drop = False, inplace = True)
test_unu_ob.reset_index(drop = False, inplace = True)


# In[143]:


train_unu_sca = pd.concat([train_unu_ob,train_unu_scaled],axis=1)
test_unu_sca = pd.concat([test_unu_ob,test_unu_scaled],axis=1)


# In[146]:


print(test_unu_sca.shape)
test_unu_drop_na.shape


# In[147]:


print(train_unu_sca.shape)
train_unu_drop_na.shape


# In[148]:


test_unu_sca.isnull().sum()


# In[149]:


train_unu_sca.to_csv('train_unu_sca_drop_na_1013.csv')
test_unu_sca.to_csv('test_unu_sca_drop_na_1013.csv')


# In[150]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[153]:


from xgboost import XGBClassifier
from xgboost import plot_importance ## Feature Importance를 불러오기 위함
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# ##  unu_drop_na예측 , test_credit도 같이 예측

# In[192]:


final_drop_na = pd.get_dummies(train_unu_sca,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])


# In[257]:


final_loan = pd.get_dummies(train_unu_loan_sca,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])


# In[273]:


final_age = pd.get_dummies(train_unu_age_sca,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])


# In[258]:


final_loan


# In[ ]:


# loan
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=777)
for train_idx, test_idx in split.split(final_drop_na, final_drop_na["is_applied"]):
    tr = final_drop_na.loc[train_idx]
    val = final_drop_na.loc[test_idx]


# In[193]:


# drop_na
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=777)
for train_idx, test_idx in split.split(final_drop_na, final_drop_na["is_applied"]):
    tr = final_drop_na.loc[train_idx]
    val = final_drop_na.loc[test_idx]


# In[194]:


print(tr["is_applied"].value_counts() / len(tr))
val["is_applied"].value_counts() / len(val)


# In[195]:


x_train


# In[196]:


x_train=tr.drop(['is_applied','houseown_type_자가', 'application_id','level_0','index'], axis=1)
y_train=tr['is_applied']


# In[198]:


x_val=val.drop(['is_applied','houseown_type_자가', 'application_id','level_0','index'], axis=1)
y_val=val['is_applied']


# In[199]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(x_train, y_train)
print("SMOTE 적용 전 학습용 피처/레이블 데이터 세트 : ", x_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트 :', X_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 값의 분포 :\n',pd.Series(y_train_over).value_counts() )


# In[263]:


final_loan.columns


# In[262]:


final_loan.drop(['index'],axis=1,inplace=True)


# In[264]:


loan_x=final_loan.drop(['is_applied','houseown_type_자가', 'application_id'], axis=1)
loan_y=final_loan['is_applied']


# In[285]:


age_x.columns


# In[ ]:


final_loan.drop(['index'],axis=1,inplace=True)


# In[287]:


age_x.drop(['index'],axis=1,inplace=True)


# In[288]:


age_x.columns


# In[289]:


age_x=final_age.drop(['is_applied','houseown_type_자가', 'application_id'], axis=1)
age_y=final_age['is_applied']


# In[267]:


final_loan.is_applied.unique()


# In[200]:


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


# In[204]:


xgb1.fit(X_train_over, y_train_over)


# In[290]:


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


# In[291]:


xgb_age.fit(age_x, age_y)


# In[265]:


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


# In[266]:


xgb_loan.fit(x_train, y_train)


# In[205]:


pred = xgb1.predict(x_val)


# In[206]:


print(classification_report(y_val, pred, target_names=['class 0', 'class 1']))


# In[208]:


test_unu_sca.columns


# In[211]:


test_unu_sca.drop(['index'], axis=1, inplace=True)


# In[213]:


## 더미변수 생성
test=pd.get_dummies(test_unu_sca, columns=['income_type', 'employment_type', 'houseown_type', 'purpose'])


# In[271]:


## 더미변수 생성
test_age=pd.get_dummies(test_unu_age_sca, columns=['income_type', 'employment_type', 'houseown_type', 'purpose'])


# In[272]:


test_age.drop(['is_applied','houseown_type_자가','application_id'],axis=1,inplace=True)


# In[221]:


test_x=test.drop(['is_applied','houseown_type_자가','application_id'], axis=1)


# In[247]:


## 더미변수 생성
test_cr=pd.get_dummies(test_unu_credit_sca, columns=['income_type', 'employment_type', 'houseown_type', 'purpose'])


# In[223]:


test_x=test_x.drop(['birth_year','houseown_type_배우자'], axis=1)


# In[248]:


test_cr.drop(['is_applied','houseown_type_자가','application_id','birth_year','houseown_type_배우자'],axis=1,inplace=True)


# In[249]:


test_cr.drop(['index'],axis=1,inplace=True)


# In[217]:


x_train.columns


# In[216]:


test_x.columns


# In[220]:


test_x.columns


# In[240]:


test_cr.columns


# In[227]:


xgb1_pred=xgb1.predict(test_x)
xgb1_pred


# In[250]:


xgb_cr_pred=xgb1.predict(test_cr)
xgb_cr_pred


# In[228]:


a1 = pd.DataFrame(xgb1_pred)
a1.tail()


# In[251]:


a2 = pd.DataFrame(xgb_cr_pred)
a2.tail()


# In[229]:


test['is_applied']=xgb1_pred
test.tail()


# In[252]:


test


# In[255]:


test_unu_drop_na.shape


# In[243]:


test_cr['is_applied']=xgb_cr_pred
test_cr.tail()


# In[230]:


test.is_applied.value_counts()


# In[ ]:


test_unu_


# In[246]:


test_unu_credit_sca


# In[244]:


test_cr.is_applied.value_counts()


# In[269]:


test_unu_loan.is_applied.fillna(1)


# In[270]:


test_unu_loan.to_csv('unu_loan_예측완_1013.csv')


# In[232]:


test.to_csv('unu_drop_na_xg_예측완_1013.csv')


# In[256]:


test_cr.to_csv('unu_cr_xg_예측완_1013.csv')


# In[ ]:


final_age = pd.get_dummies(train_unu_age_sca,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=777)
for train_idx, test_idx in split.split(final_drop_na, final_drop_na["is_applied"]):
    tr = final_drop_na.loc[train_idx]
    val = final_drop_na.loc[test_idx]

