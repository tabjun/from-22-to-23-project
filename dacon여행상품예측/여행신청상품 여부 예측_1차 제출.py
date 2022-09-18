# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:54:48 2022

@author: yoontaejun
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.stats.multicomp import MultiComparison
import scipy.stats
#from pycaret.classification import *
from sklearn.preprocessing import RobustScaler
#%%
# 플랏 한글
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#%%
# 경로 지정
os.chdir('C:\\Users\\user\\Desktop\\경진대회\\데이콘\\여행상품신청 여부 예측\\data')
print(os.getcwd())
#%%
# 데이터 불러오기 및 변수명 소문자 변경
tr = pd.read_csv('train.csv')
tr.columns = tr.columns.str.lower()
print(tr.info())
#%%
# 결측치 확인을 위한 그래프
print(tr.isnull().sum())
missing = tr.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(figsize = (12,6))
plt.show()
#%%
# 범주형 자료 값 개수 세기
print(tr['typeofcontact'].value_counts())
print(tr['citytier'].value_counts())
print(tr['occupation'].value_counts())
print(tr['gender'].value_counts())
print(tr['productpitched'].value_counts())
print(tr['preferredpropertystar'].value_counts())
print(tr['maritalstatus'].value_counts())
print(tr['passport'].value_counts())
print(tr['pitchsatisfactionscore'].value_counts())
print(tr['owncar'].value_counts())
print(tr['prodtaken'].value_counts())
#%%
# 데이터 셋 전체의 기술통계량 확인
des_tr = tr.describe(include='all')
#%%
#durationofpitch 확인
sns.distplot(tr['durationofpitch'])
plt.title('durationofpitch 플랏')
plt.show()
#%%
sns.distplot(tr['durationofpitch'])
plt.xlim(0,20)
plt.show()
#%%
# 5부터 시작 , 어차피 이산형 값으로 묶여있으니 범주로 보든지, 이산화 해주는게 좋을듯
print(tr['durationofpitch'].describe())
#%%
# lambda 적용으로 durationofpitch범주형 열 추가하기
# 이러면 결측치 처리할 때 최빈값으로 채워주던지, 아니면 각 값에 따른 다른 변수 값 보고 결측치 뭐로 채워줄지 결정하기 쉬움
tr['group_pitch_10'] = tr['durationofpitch'].apply(lambda x: '5이상 15미만' if 5 <= x < 15 else '15이상 25미만' if 15 <= x < 25 else '25이상 35미만' if 25 <= x < 35 else '35이상' if 35 <= x else x) 
print(tr)
#tr['group_pitch_5'] = tr['durationofpitch'].apply(lambda x: '5이상 10미만' if 5 <= x < 10 else '10이상 15미만' if 10 <= x < 15 else '15이상 20미만' if 15 <= x < 20 else '20이상 25미만' if 20 <= x < 25 else '25이상 30미만' if 25 <= x < 30 else '30이상 35미만' if 30 <= x < 35  else '35이상' if 35 <= x else x) 
#print(tr)
#%%
#결측치는 결측치로 잘들어갔는지 확인
print('구간_10')
print(tr[tr['group_pitch_10'].isnull()==True]['durationofpitch'].head())
#print('구간_5')
#print(tr[tr['group_pitch_5'].isnull()==True]['durationofpitch'].head())
#%%
print('구간_10')
print(tr[tr['group_pitch_10'] == '5이상 15미만']['durationofpitch'].head())
#print('구간_5')
#print(tr[tr['group_pitch_5'] == '5이상 10미만']['durationofpitch'].head())
#%%
print('구간_10')
print(tr[tr['group_pitch_10'] == '15이상 25미만']['durationofpitch'].head())
#print('구간_5')
#print(tr[tr['group_pitch_5'] == '10이상 15미만']['durationofpitch'].head())
#%%
print('구간_10')
print(tr[tr['group_pitch_10']=='25이상 35미만']['durationofpitch'].head())
#print('구간_5')
#print(tr[tr['group_pitch_5']=='15이상 20미만']['durationofpitch'].head())
#%%
#print('구간_5')
#print(tr[tr['group_pitch_5']=='20이상 25미만']['durationofpitch'].head())
#print(tr[tr['group_pitch_5']=='25이상 30미만']['durationofpitch'].head())
#print(tr[tr['group_pitch_5']=='30이상 35미만']['durationofpitch'].head())
#%%
print('구간_10')
print(tr[tr['group_pitch_10']=='35이상']['durationofpitch'].head())
#print('구간_5')
#print(tr[tr['group_pitch_5']=='35이상']['durationofpitch'].head())
# 모든 값들이 잘 바뀜을 확인
#%%
sns.countplot(x='group_pitch_10',data=tr)
#sns.countplot(x='group_pitch_5',data=tr)
plt.show()
print(tr['group_pitch_10'].value_counts())
#print(tr['group_pitch_5'].value_counts())

#%%
# durationofpitch결측치 처리
tr['group_pitch_10'] = tr['group_pitch_10'].fillna('5이상 15미만')
print(f"pitch결측치:{tr['group_pitch_10'].isnull().sum()}");print('\t')
print(tr['group_pitch_10'].value_counts())
#%%
# 원래 열 없애기
tr = tr.drop(['durationofpitch'],axis=1)
#%%
#preferredpreopertystar 빈도 수 확인
sns.countplot(x='preferredpropertystar',data=tr)
plt.show()
print(tr['preferredpropertystar'].value_counts())
print(tr['preferredpropertystar'].isnull().sum())
#%%
#preferredpreopertystar 결측치 확인
tr['preferredpropertystar'] = tr['preferredpropertystar'].fillna(3.0)
print(f"pitch결측치:{tr['preferredpropertystar'].isnull().sum()}");print('\t')
print(tr['preferredpropertystar'].value_counts())
print(tr['preferredpropertystar'].isnull().sum())
#%%
#여행횟수 빈도 확인
sns.countplot(x='numberoftrips',data=tr)
plt.show()
print(tr['numberoftrips'].value_counts())
print(f"결측치:{tr['numberoftrips'].isnull().sum()}")
#%%
#여행횟수 결측치 처리
tr['numberoftrips'] = tr['numberoftrips'].fillna(2.0)
print(tr['numberoftrips'].value_counts())
print(tr['numberoftrips'].isnull().sum())
#%%
# monthlyincome 오른쪽 꼬리라서 로그화해줄 필요가 있음. 근데 결측치 채우고 해야하니까
# 뒤에 할 groupby나 피벗테이블을 통해 대략의 값을 파악하여 채워줄 예정
sns.distplot(tr['monthlyincome'])
plt.show()
print(tr['monthlyincome'].describe())
#%%
# 결측치를 제외하고 보기 위한 데이터 추출
non_null = tr[tr['monthlyincome'].isnull()==False]
#%%
# 분산분석으로 직책이 수입에 영향을 끼치는지 검정, 일단 가정은 건너뜀
model = ols('monthlyincome ~ designation', non_null).fit()
print(anova_lm(model))
#%%
# 분산분석으로 직책이 수입에 영향을 끼치는지 검정, 일단 가정은 건너뜀
model = ols('monthlyincome ~ typeofcontact', non_null).fit()
print(anova_lm(model))
#%%
# 성별에 따른 임금차이
model = ols('monthlyincome ~ gender', non_null).fit()
print(anova_lm(model))
#%%
# 직업에 따른 임금 차이, 차이없
model = ols('monthlyincome ~ occupation', non_null).fit()
print(anova_lm(model))
#%%
# 여행횟수에 따른 임금차이
model = ols('monthlyincome ~ numberoftrips', non_null).fit()
print(anova_lm(model))
#%%
#차 여부
model = ols('monthlyincome ~ owncar', non_null).fit()
print(anova_lm(model))
#%%
# 식 도출했을 때 모든 변수 유의함
model = ols('monthlyincome ~ owncar + numberoftrips + gender + designation', non_null).fit()
print(anova_lm(model))
#%%
# 직책에 따른 수입, 사후분석
a = MultiComparison(non_null['monthlyincome'], non_null['designation'])
print(a)
# 봉페르니를 이용한 사후분석
result = a.allpairtest(scipy.stats.ttest_ind, method='bonf')
print(result[0])
#%%
# 여행횟수별, 수입에 결측치가 존재하는 직급 보기
print(tr[(tr['monthlyincome'].isnull()==True)&(tr['numberoftrips']==1)]['designation'].value_counts())
print(tr[(tr['monthlyincome'].isnull()==True)&(tr['numberoftrips']==2)]['designation'].value_counts())
print(tr[(tr['monthlyincome'].isnull()==True)&(tr['numberoftrips']==3)]['designation'].value_counts())
print(tr[(tr['monthlyincome'].isnull()==True)&(tr['numberoftrips']==4)]['designation'].value_counts())
print(tr[(tr['monthlyincome'].isnull()==True)&(tr['numberoftrips']==5)]['designation'].value_counts())
print(tr[(tr['monthlyincome'].isnull()==True)&(tr['numberoftrips']==6)]['designation'].value_counts())
print(tr[(tr['monthlyincome'].isnull()==True)&(tr['numberoftrips']==7)]['designation'].value_counts())
print(tr[(tr['monthlyincome'].isnull()==True)&(tr['numberoftrips']==8)]['designation'].value_counts())
#%%
# 앞서 구한 그룹별 평균으로 결측치 대체
tr['monthlyincome'] = tr['monthlyincome'].fillna(tr.groupby(['owncar',
                                                             'gender',
                                                             "designation",
                                                             'numberoftrips'])['monthlyincome'].transform('mean'))
#%%
#거의 정규분포 형태라서 별 다르게 로그화 해줄 필요는 없을 듯
sns.distplot(tr['age'])
plt.show()
#%%
# age결측치 채워주기위한 데이터 분리
tr_copy = tr.copy()
age_null = tr_copy[tr_copy['age'].isnull()==True]
print(age_null)
#%%
age_non = tr_copy[tr_copy['age'].isnull()==False]
#%%
#결측치 채우기
sns.countplot(x='typeofcontact',data=tr)
plt.show()
print(tr['typeofcontact'].value_counts())
#%%
#약 2.5배차이, 최빈값으로 채워넣기
tr['typeofcontact'] = tr['typeofcontact'].fillna('Self Enquiry')
print(tr['typeofcontact'].value_counts())
print(tr['typeofcontact'].isnull().sum())
#%%
sns.countplot(x='numberoffollowups',data=tr)
plt.show()
print(tr['numberoffollowups'].value_counts())
print(tr['numberoffollowups'].isnull().sum())
#%%
tr['numberoffollowups'] = tr['numberoffollowups'].fillna(4.0)
print(tr['numberoffollowups'].value_counts())
print(tr['numberoffollowups'].isnull().sum())
#%%
sns.countplot(x='numberofchildrenvisiting',data=tr)
plt.show()
print(tr['numberofchildrenvisiting'].value_counts())
print(tr['numberofchildrenvisiting'].isnull().sum())
#%%
tr['numberofchildrenvisiting'] = tr['numberofchildrenvisiting'].fillna(1.0)
print(tr['numberofchildrenvisiting'].value_counts())
print(tr['numberofchildrenvisiting'].isnull().sum())
#%%
#차 여부
model = ols('age ~ owncar', age_non).fit()
print(anova_lm(model))
#%%
# 도시 계급, 유의하지 않음
model = ols('age ~ citytier', age_non).fit()
print(anova_lm(model))
#%%
# 직업에 따른 나이, 유의하지 않음
model = ols('age ~ occupation', age_non).fit()
print(anova_lm(model))
#%%
# 여행추천상품에 따른 나이
model = ols('age ~ productpitched', age_non).fit()
print(anova_lm(model))
#%%
# 직업에 따른 나이, 유의하지 않음
model = ols('age ~ gender', age_non).fit()
print(anova_lm(model))
#%%
# 선호호텔 등급 따른 나이, 유의하지 않음
model = ols('age ~ preferredpropertystar', age_non).fit()
print(anova_lm(model))
#%%
# 결혼상태에 따른 나이
model = ols('age ~ maritalstatus', age_non).fit()
print(anova_lm(model))
#%%
# 여행횟수에 따른 나이
# 회귀분석으로 돌리나 이걸로 돌리나 같은 결과로 나옴
model = ols('age ~ numberoftrips', age_non).fit()
print(anova_lm(model))
#%%
model = ols('age ~ numberoftrips', age_non).fit()                                                                                
print(anova_lm(model))
#%%
# 여권 여부에 따른 나이, 유의하지 않음
model = ols('age ~ passport', age_non).fit()
print(anova_lm(model))
#%%
# 직업에 따른 나이, 유의하지 않음
model = ols('age ~ pitchsatisfactionscore', age_non).fit()
print(anova_lm(model))
#%%
# 직업에 따른 나이, 유의하지 않음
model = ols('age ~ numberofchildrenvisiting', age_non).fit()
print(anova_lm(model))
#%%
# 직급에 따른 나이
model = ols('age ~ designation', age_non).fit()
print(anova_lm(model))
#%%
# 직책에 따른 나이, 사후분석
a = MultiComparison(age_non['age'], age_non['designation'])
print(a)
# vp랑 avp가 차이가 없음 그럼 다른거랑 묶어서 같다로 넣어줘야하는데 일단 나중에
# 봉페르니를 이용한 사후분석
result = a.allpairtest(scipy.stats.ttest_ind, method='bonf')
print(result[0])
#%%
model = ols('age ~ designation + maritalstatus + owncar + numberoftrips + productpitched', age_non).fit()
print(anova_lm(model))
#%%
tr['age'] = tr['age'].fillna(tr.groupby(['owncar','numberoftrips',
                                         'maritalstatus','productpitched'])['age'].transform('mean'))
#%%
tr['age'] = round(tr['age'],0)
#%%
print(tr.isnull().sum())
#%%
sns.distplot(tr.age)
plt.show()
#%%
sns.distplot(tr.monthlyincome)
plt.show()
#%%
tr_copy = tr.copy()
tr_1 = tr_copy[tr_copy['numberoftrips']==1]
tr_2 = tr_copy[tr_copy['numberoftrips']==2]
tr_3 = tr_copy[tr_copy['numberoftrips']==3]
tr_4 = tr_copy[tr_copy['numberoftrips']==4]
tr_5 = tr_copy[tr_copy['numberoftrips']==5]
tr_6 = tr_copy[tr_copy['numberoftrips']==6]
tr_7 = tr_copy[tr_copy['numberoftrips']==7]
tr_8 = tr_copy[tr_copy['numberoftrips']==8]
tr_19 = tr_copy[tr_copy['numberoftrips']==19]
#%%
des_1 = tr_1.describe(include='all')
des_2 = tr_2.describe(include='all')
des_3 = tr_3.describe(include='all')
des_4 = tr_4.describe(include='all')
des_5 = tr_5.describe(include='all')
des_6 = tr_6.describe(include='all')
des_7 = tr_7.describe(include='all')
des_8 = tr_8.describe(include='all')
des_19 = tr_19.describe(include='all')
print(des_19)
#%%
# 여행횟수 이상치 3으로 변경
tr.loc[tr['numberoftrips'] == 19] = 3
print(tr.loc[tr['numberoftrips'] == 19])
#%%
# 데이터 프레임 내 변수 박스플랏 그리기
f, ax = plt.subplots(figsize=(16, 14))
ax.set_xscale("log")
ax = sns.boxplot(data = tr , orient="h", palette="Set1")

ax.xaxis.grid(False)

plt.xlabel("Numeric values", fontsize = 10)
plt.ylabel("Feature names", fontsize = 10)
plt.title("Numeric Distribution of Features", fontsize = 15)
sns.despine(trim = True, left = True)
# numberofpersonvisiting, numberoffollowups, numberoftripsm, montlyincome이상치 존재
# 일단 로버스트 정규화로 돌려보기
#%%
#라벨 인코딩을 위한 문자형 변수만 추출
object_columns = tr.columns[tr.dtypes == 'object']
print('object 칼럼은 다음과 같습니다 : ', list(object_columns))
#%%
train_enc = tr.copy()
# train 문자형 변수 라벨 인코더
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])
#%%
print(train_enc.dtypes)
print(train_enc)
#%%
# 로버스트 스케일러
#transformer = RobustScaler()
# ,normalize = True,normalize_method='robust',,imputation_type='iterative'
#tr[['monthlyincome','age']]=transformer.fit(tr[['monthlyincome','age']])
#%%
exp_clf = setup(data = tr, target = 'prodtaken',
                fold_shuffle = True,
                session_id=1234,ignore_features=['id'])
#%%
#모델 형성 및 성능 비교
top10_model = compare_models(sort='Accuracy',fold=5,n_select = 10)
print(top10_model)
top5_model_f1 = compare_models(fold=5, n_select = 5, sort='F1')
print(top5_model_f1)
top5_model_prec = compare_models(fold=5, n_select = 5, sort='Precision')
print(top5_model_prec)

#%%
# 테스트 셋
test = pd.read_csv('test.csv')
test.columns = test.columns.str.lower()
print(test.info())
print(test.isnull().sum())
#%%
test['group_pitch_10'] = test['durationofpitch'].apply(lambda x: '5이상 15미만' if 5 <= x < 15 else '15이상 25미만' if 15 <= x < 25 else '25이상 35미만' if 25 <= x < 35 else '35이상' if 35 <= x else x) 
#%%
test['monthlyincome'] = test['monthlyincome'].fillna(test.groupby(['owncar','gender',"designation",'numberoftrips'])['monthlyincome'].transform('mean'))
#%%
test['age'] = test['age'].fillna(test.groupby(['owncar',"designation",'numberoftrips','maritalstatus','productpitched'])['age'].transform('mean'))
#%%
#결측치 채우기
sns.countplot(x='typeofcontact',data=test)
plt.show()
print(test['typeofcontact'].value_counts())
#%%
#약 2.5배차이, 최빈값으로 채워넣기
test['typeofcontact'] = test['typeofcontact'].fillna('Self Enquiry')
print(tr['typeofcontact'].isnull().sum())
#%%
sns.countplot(x='numberofchildrenvisiting',data=test)
plt.show()
print(test['numberofchildrenvisiting'].value_counts())
print(test['numberofchildrenvisiting'].isnull().sum())
#%%
test['numberofchildrenvisiting'] = test['numberofchildrenvisiting'].fillna(1.0)
print(test['numberofchildrenvisiting'].isnull().sum())
#%%
sns.countplot(x='numberoffollowups',data=test)
plt.show()
print(test['numberoffollowups'].value_counts())
print(test['numberoffollowups'].isnull().sum())
#%%
test['numberoffollowups'] = test['numberoffollowups'].fillna(4.0)
print(test['numberoffollowups'].isnull().sum())
#%%
test['age'] = round(test['age'],0)
#%%
#preferredpreopertystar 빈도 수 확인
sns.countplot(x='preferredpropertystar',data=test)
plt.show()
print(test['preferredpropertystar'].value_counts())
print(test['preferredpropertystar'].isnull().sum())
#%%
#preferredpreopertystar 결측치 확인
test['preferredpropertystar'] = test['preferredpropertystar'].fillna(3.0)
print(f"pitch결측치:{test['preferredpropertystar'].isnull().sum()}");print('\t')
print(test['preferredpropertystar'].value_counts())
print(test['preferredpropertystar'].isnull().sum())
#%%
#여행횟수 빈도 확인
sns.countplot(x='numberoftrips',data=test)
plt.show()
print(test['numberoftrips'].value_counts())
print(f"결측치:{test['numberoftrips'].isnull().sum()}")
#%%
#여행횟수 결측치 처리
test['numberoftrips'] = test['numberoftrips'].fillna(2.0)
print(test['numberoftrips'].isnull().sum())
#%%
test = test.drop(['durationofpitch'],axis=1)
#%%
print(test.isnull().sum())
#%%
print(test['numberoftrips'].unique())
print(test['numberoftrips'].value_counts())
#%%
print(test[test['numberoftrips']==2])
print(test[test['numberoftrips']==20])
print(test[test['numberoftrips']==21])
print(test[test['numberoftrips']==22])
#%%
test.loc[test['numberoftrips'] == 20] = 3
test.loc[test['numberoftrips'] == 21] = 3
test.loc[test['numberoftrips'] == 22] = 3
print(test.loc[test['numberoftrips'] == 20])

#%%
# 모델 생성
ct = create_model('catboost')
rf = create_model('rf')
et = create_model('et')
xb = create_model('xgboost')
#%%
#모델 튜닝
tuned_rf = tune_model(rf)
tuned_ct = tune_model(ct)
tuned_et = tune_model(et)
tuned_xb = tune_model(xb)
#%%
#캣부스트 적용 f1스코어 1등
final_ct = finalize_model(tuned_ct)
pred_ct = predict_model(final_ct, data = test)
#%%
# 랜포 prec 1등
final_rf = finalize_model(tuned_rf)
pred_rf = predict_model(final_rf, data = test)
#%%
# 엑스트라 트리 정확도 1등
final_et = finalize_model(tuned_et)
#%%
final_xb = finalize_model(tuned_xb)
pred_xb = predict_model(final_xb, data = test)
#%%
pred_xb.to_csv('C:/Users/user\Desktop/경진대회/데이콘/여행상품신청 여부 예측/data/xb.csv',
               columns = ['id','Label'])