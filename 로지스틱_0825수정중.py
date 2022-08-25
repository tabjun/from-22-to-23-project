# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:00:00 2022

@author: 215-05
"""

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # train과 validation data를 분리해줌.
from sklearn import linear_model  # logistic regression model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
#%%
# 경로 지정
os.chdir('C:\\Users\\215-05\\Downloads')
print(os.getcwd())
#%%
# 데이터 불러오기 및 변수명 소문자 변경
tr = pd.read_csv('train.csv')
tr.columns = tr.columns.str.lower()
print(tr.info())
#%%
print((tr.isnull().any(axis=1)).value_counts())
print(tr.isnull().any(axis=0))
tr_null = tr.copy().dropna(axis=0)
print(tr_null.info())
#%%
logis = sm.Logit.from_formula('prodtaken ~ age', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
logis = sm.Logit.from_formula('prodtaken ~ citytier', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
logis = sm.Logit.from_formula('prodtaken ~ numberofpersonvisiting', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
logis = sm.Logit.from_formula('prodtaken ~ numberoffollowups', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
logis = sm.Logit.from_formula('prodtaken ~ preferredpropertystar', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
logis = sm.Logit.from_formula('prodtaken ~ pitchsatisfactionscore', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
# 유의하지 않음
logis = sm.Logit.from_formula('prodtaken ~ numberofchildrenvisiting', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
logis = sm.Logit.from_formula('prodtaken ~ monthlyincome', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
logis = sm.Logit.from_formula('prodtaken ~ productpitched', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
#유의하지 않음
logis = sm.Logit.from_formula('prodtaken ~ owncar', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
#유의하지 않음
logis = sm.Logit.from_formula('prodtaken ~ numberoftrips', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
logis = sm.Logit.from_formula('prodtaken ~ typeofcontact', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
#모델은 유의하나 계수는 유의하지 않음
logis = sm.Logit.from_formula('prodtaken ~ occupation', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
#유의하지 않음
logis = sm.Logit.from_formula('prodtaken ~ gender', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
#Married유의하지 않음
logis = sm.Logit.from_formula('prodtaken ~ maritalstatus', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
#vp,manager,senior manager유의하지 않음
logis = sm.Logit.from_formula('prodtaken ~ designation', tr_null).fit()
print(logis.summary())
np.exp(logis.params)
#%%
#조건에 따른 행 제거
des = tr_null.copy()
index1 = des[des['designation']=='VP'].index
des=des.drop(index1)
print(des['designation'].value_counts())
#%%
logis = sm.Logit.from_formula('prodtaken ~ designation', des).fit()
print(logis.summary())
#%%
des = tr_null.copy()
index1 = des[des['designation']=='Senior Manager'].index
des=des.drop(index1)
print(des['designation'].value_counts())
#%%
logis = sm.Logit.from_formula('prodtaken ~ designation', des).fit()
print(logis.summary())
#%%
des = tr_null.copy()
index1 = des[des['designation']=='Manager'].index
des=des.drop(index1)
print(des['designation'].value_counts())
#%%
logis = sm.Logit.from_formula('prodtaken ~ designation', des).fit()
print(logis.summary())
#%%
des = tr_null.copy()
index1 = des[(des['designation']=='VP')|(des['designation']=='Manager')].index
des=des.drop(index1)
print(des['designation'].value_counts())
#%%
logis = sm.Logit.from_formula('prodtaken ~ designation', des).fit()
print(logis.summary())
#%%
des = tr_null.copy()
index1 = des[(des['designation']=='VP')|(des['designation']=='Senior Manager')].index
des=des.drop(index1)
print(des['designation'].value_counts())
#%%
logis = sm.Logit.from_formula('prodtaken ~ designation', des).fit()
print(logis.summary())
#%%
des = tr_null.copy()
index1 = des[(des['designation']=='Senior Manager')|(des['designation']=='Manager')].index
des=des.drop(index1)
print(des['designation'].value_counts())
#%%
logis = sm.Logit.from_formula('prodtaken ~ designation', des).fit()
print(logis.summary())
#%%
# 데이터 프레임 내 변수 박스플랏 그리기
f, ax = plt.subplots(figsize=(16, 14))
ax.set_xscale("log")
ax = sns.boxplot(data = tr_null , orient="h", palette="Set1")

ax.xaxis.grid(False)

plt.xlabel("Numeric values", fontsize = 10)
plt.ylabel("Feature names", fontsize = 10)
plt.title("Numeric Distribution of Features", fontsize = 15)
sns.despine(trim = True, left = True)
#%%
tr_null_x = tr_null.drop(['id','prodtaken'],axis=1)
tr_null_y = tr_null['prodtaken']
#%%
cols = tr_null_x.select_dtypes(exclude=['object']).columns

scaler = RobustScaler()
#StandardScaler()도 고려
# 학습용 데이터를 이용해 scaler를 학습시킵니다.
scaler.fit(tr_null_x[cols])
# 학습된 scaler를 사용해 데이터를 변환합니다.
scaled = scaler.transform(tr_null_x[cols])

# 변환된 값을 새로운 column에 할당합니다.
tr_null_x[cols] = scaled

#%%
tr_x = pd.get_dummies(tr_null_x)
#%%
m_tr = pd.concat([tr_x, tr_null_y], axis = 1)
#%%
train_x, valid_x, train_y, valid_y = train_test_split(tr_x, tr_null_y, test_size=0.2,random_state=1234)
print(train_x.shape, valid_x.shape, train_y.shape, valid_y.shape)
#%%
model = sm.Logit(train_y, train_x)
results = model.fit(method = "newton")
results.summary()
#%%
#열 제거안하고 모델링한다음 비교
x = tr_null.drop['id','prodtaken']
y = tr_null['prodtaken']
#%%
object_columns = train.columns[train.dtypes == 'object']
print('object 칼럼은 다음과 같습니다 : ', list(object_columns))

# 해당 칼럼만 보아서 봅시다
train[object_columns]
#%%
train_enc = train_nona.copy()

# 모든 문자형 변수에 대해 encoder를 적용합니다.
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])

# 결과를 확인합니다.
train_enc
