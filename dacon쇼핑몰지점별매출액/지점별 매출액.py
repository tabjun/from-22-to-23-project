# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:32:09 2022

@author: yoontaejun
"""

#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import kruskal
from scipy.stats import bartlett
from scipy.stats import levene
import scipy.stats
from scipy.stats import wilcoxon
import scipy.stats as stats
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
#%%
os.chdir('C:\\Users\\user\\Desktop\\경진대회\\데이콘\\쇼핑몰 지점별 매출액 예측')
print(os.getcwd())
#%%
# 플랏 한글
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#%%
#숫자 지수표현없이 출력
pd.options.display.float_format = '{:.2f}'.format
#%%
df = pd.read_csv('train.csv')
df.columns = df.columns.str.lower()
df_copy = df.copy()
#%%
print(df.info())
#%%
print(test.info())
#%%
print(df.head())
#%%
print(df.columns)
#%%
df_copy = df.copy()
df['date'] = pd.to_datetime(df['date'],format='%d/%m/%Y')
df = df.astype({'store':'category'})
print(df.dtypes)
#%%
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
print(df)
#%%
# 범례 제거, 날짜에 따른 지점별 매출 추이
plt.figure(figsize=(10,7))
sns.lineplot(data=df,x='date',y='weekly_sales',ci=None,hue='store',legend='full')
plt.legend([],[], frameon=False) # 범례 제거
plt.show()
print('일정기간 동안 매출이 급증함')
#%%
df_prom = df[(df['promotion1'].isnull()==False)|(df['promotion2'].isnull()==False)|(df['promotion3'].isnull()==False)|(df['promotion4'].isnull()==False)|(df['promotion5'].isnull()==False)]
print(df_prom)
#%%
df_null = df[(df['promotion1'].isnull()==True)&(df['promotion2'].isnull()==True)&(df['promotion3'].isnull()==True)&(df['promotion4'].isnull()==True)&(df['promotion5'].isnull()==True)]
print(df_null)
#%%
# 0이 있으면 흑자 적자 일치하는 날
print(f'promotion1:{len(df_prom[df_prom["promotion1"] == 0])}')
print(f'promotion2:{len(df_prom[df_prom["promotion2"] == 0])}')
print(f'promotion3:{len(df_prom[df_prom["promotion3"] == 0])}')
print(f'promotion4:{len(df_prom[df_prom["promotion4"] == 0])}')
print(f'promotion5:{len(df_prom[df_prom["promotion5"] == 0])}')
print('''prom2,3에 0이 있음 이는 이익과 손실이 일치하는 날, 이를 통해 결측치는 프로모션을 진행하지
      않은 기간이라고 설정가능함''')      
#%%
# promotion 시각화
sns.distplot(df_prom['promotion1'],color="blue", label="promotion1")
sns.distplot(df_prom['promotion2'],color="red", label="promotion2")
sns.distplot(df_prom['promotion3'],color="green", label="promotion3")
sns.distplot(df_prom['promotion4'],color="yellow", label="promotion4")
sns.distplot(df_prom['promotion5'],color="gray", label="promotion5")
plt.legend(title="protion")
plt.title('Histogram & Density Plot by Groups')
plt.legend(prop={'size': 12}, title = 'Group')
plt.show()
#%%
multi_plot= {'promotion1': 'blue',
             'promotion2': 'red',
             'promotion3': 'yellow',
             'promotion4': 'green',
             'promotion5': 'violet',}
                  
i = [0, 0, 1, 1,2]
j = [0, 1, 0, 1,0]
f,axes = plt.subplots(3, 2, figsize=(8, 6), sharex=True)
for var, i, j in zip(multi_plot, i, j):
    sns.distplot(df_prom[var],
                 color = multi_plot[var],
                 ax= axes[i,j],)
plt.show()
plt.tight_layout()
print('오른쪽 긴 꼬리')
#%%
store_1 = df[(df['store']==1)&(df['date']<='2011-02-01')]
store_1_1 = df[(df['store']==1)&(('2011-02-01'<=df['date'])&(df['date']<='2012-02-01'))]
#%%
# 범례 제거, 날짜에 따른 지점별 매출 추이
plt.figure(figsize=(10,7))
sns.lineplot(data=store_1,x='date',y='weekly_sales',ci=None,hue='store',legend='full')
plt.show()
print('일정기간 동안 매출이 급증함')
#%%
plt.figure(figsize=(10,7))
sns.lineplot(data=store_1_1,x='date',y='weekly_sales',ci=None,hue='store',legend='full')
plt.show()
print('일정기간 동안 매출이 급증함')
#%%
s_45 = df[df['store']==45]
plt.figure(figsize=(10,7))
sns.lineplot(data=s_45,x='date',y='weekly_sales',ci=None,hue='store',legend='full')
plt.show()
#%%
#세부비교
dd = df[(df['store']==1)&(('2011-09-01'<=df['date'])&(df['date']<='2011-12-01'))]
plt.figure(figsize=(10,7))
sns.lineplot(data=dd,x='date',y='weekly_sales',ci=None,hue='store',legend='full')
plt.show()
#%%
#세부비교
dd = df[(df['store']==1)&(('2011-11-01'<=df['date'])&(df['date']<='2011-12-01'))]
plt.figure(figsize=(10,7))
sns.lineplot(data=dd,x='date',y='weekly_sales',ci=None,hue='store',legend='full')
plt.show()
#%%
dd = df[(df['store']==1)&(('2011-12-01'<=df['date'])&(df['date']<='2012-01-01'))]
plt.figure(figsize=(10,7))
sns.lineplot(data=dd,x='date',y='weekly_sales',ci=None,hue='store',legend='full')
plt.show()
#%%
dd = df[(df['store']==1)&(('2012-01-01'<=df['date'])&(df['date']<='2012-02-01'))]
plt.figure(figsize=(10,7))
sns.lineplot(data=dd,x='date',y='weekly_sales',ci=None,hue='store',legend='full')
plt.show()
#%%
print(shapiro(df_null['weekly_sales']))
print(shapiro(df_prom['weekly_sales']))
# 정규성 만족 안함
print(levene(df_null['weekly_sales'],df_prom['weekly_sales']))
#%%
#개수를 맞추기 위한 랜덤 추출
s_null = df_null.sample(n=2115)
#%%
# 대응표본 t검정 
# 정규성을 만족하지 못하여 비모수적 방법  wilcoxon으로 중앙값 비교
print(scipy.stats.ttest_rel(s_null['weekly_sales'],df_prom['weekly_sales']))
print(wilcoxon(s_null['weekly_sales'],df_prom['weekly_sales']))
# promotion에 따른 매출차이가 없다.
#%%
df['is_holiday'] = [1 if s == True else 0 for s in df['isholiday']]
print(df)
#%%
df_s1 = df[df['store']==1]
print(df_s1)
#%%
df = df.astype({'is_holiday':'category'})
#%%
print(df.info())
#%%
df= df.drop(labels='isholiday',axis=1)
print(df1.info())
#%%
print(df1['weekly_sales'].describe())
sns.distplot(df1['weekly_sales'],color="blue", label="goal")
#%%
sns.countplot(x=df1["is_holiday"])
plt.title('주말여부')
plt.show()
# 1이 너무적음
#%%
print(df['store'].value_counts)
print(df1['is_holiday'].value_counts())
#%%
print(f'test셋 휴일여부 개수:{test["IsHoliday"].value_counts()}')
#%%
df = pd.get_dummies(df,columns=['is_holiday'])
print(df.head())
#%%
#더미변수 회귀분석
model = ols('weekly_sales ~ is_holiday_0 + is_holiday_1', df1_dum).fit()                                                                                
print(anova_lm(model))
# 주말여부에 '1은 유의하지 않은 변수라고 판단 제거하는게 좋을 듯
#%%
# 수치형으로 프로모션에 대한 회귀분석
model = ols('weekly_sales ~ is_holiday_0 + is_holiday_1 + promotion1 + promotion2 + promotion3 + promotion4 + promotion5 + temperature + fuel_price + unemployment', df).fit()                                                                                
print(anova_lm(model))
#%%
df_prom['prom1'] = [1 if s == True else 0 for s in df_prom['isholiday'].isnull()]
print(df_prom['prom1'].value_counts())
#%%
# 데이터 프레임 내 변수 박스플랏 그리기
f, ax = plt.subplots(figsize=(16, 14))
ax.set_xscale("log")
ax = sns.boxplot(data = df , orient="h", palette="Set1")

ax.xaxis.grid(False)

plt.xlabel("Numeric values", fontsize = 10)
plt.ylabel("Feature names", fontsize = 10)
plt.title("Numeric Distribution of Features", fontsize = 15)
sns.despine(trim = True, left = True)
#%%
'지점별 매출 알아보기'
#%%
pdf1 = pd.pivot_table(df_prom,                # 피벗할 데이터프레임
                     index = 'store',    # 행 위치에 들어갈 열
                     columns = 'month',    # 열 위치에 들어갈 열
                     values = 'weekly_sales',     # 데이터로 사용할 열
                     aggfunc = 'sum')   # 데이터 집계함수
ppdf1 = pd.DataFrame(pdf1)  
print(ppdf1)
#%%
pdf2 = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'store',    # 행 위치에 들어갈 열
                     #columns = 'sex',    # 열 위치에 들어갈 열
                     values = 'weekly_sales',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
ppdf = pd.DataFrame(pdf2)
print(ppdf)
#%%
'romotion 지점별 편차 알아보기'
#%%
pdf2 = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'store',    # 행 위치에 들어갈 열
                     #columns = 'sex',    # 열 위치에 들어갈 열
                     values = 'weekly_sales',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
ppdf = pd.DataFrame(pdf2)
print(ppdf.sort_values('weekly_sales',ascending=False))
#%%
pdf2 = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'store',    # 행 위치에 들어갈 열
                     #columns = 'sex',    # 열 위치에 들어갈 열
                     values = 'promotion1',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
ppdf = pd.DataFrame(pdf2)
print(ppdf.sort_values('promotion1',ascending=False))

#%%
pdf2 = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'store',    # 행 위치에 들어갈 열
                     #columns = 'sex',    # 열 위치에 들어갈 열
                     values = 'promotion2',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
ppdf = pd.DataFrame(pdf2)
print(ppdf.sort_values('promotion2',ascending=False))
#%%
pdf2 = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'store',    # 행 위치에 들어갈 열
                     #columns = 'sex',    # 열 위치에 들어갈 열
                     values = 'promotion3',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
ppdf = pd.DataFrame(pdf2)
print(ppdf.sort_values('promotion3',ascending=False))
#%%
pdf2 = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'store',    # 행 위치에 들어갈 열
                     #columns = 'sex',    # 열 위치에 들어갈 열
                     values = 'promotion4',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
ppdf = pd.DataFrame(pdf2)   
print(ppdf.sort_values('promotion4',ascending=False))
#%%
pdf2 = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'store',    # 행 위치에 들어갈 열
                     #columns = 'sex',    # 열 위치에 들어갈 열
                     values = 'promotion5',     # 데이터로 사용할 열
                     aggfunc = 'mean')   # 데이터 집계함수
ppdf = pd.DataFrame(pdf2)
print(ppdf.sort_values('promotion5',ascending=False))
#%%
# 상관계수 도출을 위한 연속형 변수 추출
cor_pr = df_prom[['temperature','fuel_price','promotion1','promotion2','promotion3',
                  'promotion4','promotion5','unemployment','weekly_sales']]
null_pr = df_null[['temperature','fuel_price','promotion1','promotion2','promotion3',
                  'promotion4','promotion5','unemployment','weekly_sales']]
cor_df = df[['temperature','fuel_price','promotion1','promotion2','promotion3',
                  'promotion4','promotion5','unemployment','weekly_sales']]
#%%
mask = np.triu(cor_pr.corr())
prom_corr = cor_pr.corr()
plt.figure(figsize=(10,10))
sns.heatmap(prom_corr,annot=True,mask=mask)
plt.title('promotion진행한것만')
plt.show()
#%%
mask = np.triu(null_pr.corr())
null_corr = null_pr.corr()
plt.figure(figsize=(10,10))
sns.heatmap(null_corr,annot=True,mask=mask)
plt.title('promotion진행안한거')
plt.show()
#%%
#지점별 매출 추세 그래프
from matplotlib import dates
fig = plt.figure(figsize=(35,35)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정

for i in range(1,46):
    train2 = df[df.store == i]

    train2  = train2[["date", "weekly_sales"]]
    
    ax = fig.add_subplot(8,8,i) ## 그림 뼈대(프레임) 생성


    plt.title("store_{}".format(i)) 
    plt.ylabel('weekly_sales')
    plt.xticks(rotation=15)
    ax.xaxis.set_major_locator(dates.MonthLocator(interval = 2))
    ax.plot(train2["date"], train2["weekly_sales"],marker='',label='train', color="blue")

plt.show()
#%%
# 결측치 대체
# 값확인용
df.groupby('store').mean()

#  lambda 함수 만들고 apply
fill_mean_func = lambda x: x.fillna(x.mean())
df.groupby('store').apply(fill_mean_func)
#%%
multi_plot= {'promotion1': 'blue',
             'promotion2': 'red',
             'promotion3': 'yellow',
             'promotion4': 'green',
             'promotion5': 'violet',}
                  
i = [0, 0, 1, 1,2]
j = [0, 1, 0, 1,0]
f,axes = plt.subplots(3, 2, figsize=(8, 6), sharex=True)
for var, i, j in zip(multi_plot, i, j):
    sns.distplot(df_non_null[var],
                 color = multi_plot[var],
                 ax= axes[i,j],)
plt.show()
plt.tight_layout()
print('결측치 0 채운 플랏, 분포 높이가 바뀌어버림')
#%%
print(df1[(df1['is_holiday']==1)&((df1['promotion1'].isnull()==False)|(df1['promotion2'].isnull()==False)|(df1['promotion3'].isnull()==False)|(df1['promotion4'].isnull()==False)|(df1['promotion5'].isnull()==False))])
#%%
df['weekly_Sales'] = np.log1p(df['weekly_sales'])
#%%
df['weekly_Sales'] = np.log1p(df['weekly_sales'])
#%%
print(df[['promotion1','promotion2','promotion3','promotion4','promotion5']].describe())
#%%
df.loc[df['promotion1'] > 1] = 0.73
df.loc[df['promotion2'] > 1] = 0.45
df.loc[df['promotion5'] > 1] = 0.74
#%%
df = df.replace(df[df['promotion1'] > 1], 0.73)
df = df.replace(df[df['promotion2'] > 1], 0.45)
df = df.replace(df[df['promotion5'] > 1], 0.74)
#%%
df['promotion1'] = np.log1p(df['promotion1'])
df['promotion2'] = np.log1p(df['promotion2'])
df['promotion3'] = np.log1p(df['promotion3'])
df['promotion4'] = np.log1p(df['promotion4'])
df['promotion5'] = np.log1p(df['promotion5'])
#%%
'로그변환'
df['weekly_Sales'] = np.log1p(df['weekly_sales'])
df['fuel_price'] = np.log1p(df['fuel_price'])
df['temperature'] = np.log1p(df['temperature'])
df['unemployment'] = np.log1p(df['unemployment'])
#%%
multi_plot= {'promotion1': 'blue',
             'promotion2': 'red',
             'promotion3': 'yellow',
             'promotion4': 'green',
             'promotion5': 'violet',}
                  
i = [0, 0, 1, 1,2]
j = [0, 1, 0, 1,0]
f,axes = plt.subplots(3, 2, figsize=(8, 6), sharex=True)
for var, i, j in zip(multi_plot, i, j):
    sns.distplot(df[var],
                 color = multi_plot[var],
                 ax= axes[i,j],)
plt.show()
plt.tight_layout()
print('오른쪽 긴 꼬리')
#%%
multi_plot= {'fuel_price': 'blue',
             'weekly_sales': 'red',
             'temperature': 'yellow',
             'unemployment': 'green',}
                  
i = [0, 0, 1, 1]
j = [0, 1, 0, 1]
f,axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
for var, i, j in zip(multi_plot, i, j):
    sns.distplot(df_copy[var],
                 color = multi_plot[var],
                 ax= axes[i,j],)
plt.show()
plt.tight_layout()
print('오른쪽 긴 꼬리')
#%%
model = ols('weekly_sales ~ is_holiday_0 + is_holiday_1', df).fit()                                                                                
print(anova_lm(model))
#%%
model = ols('weekly_sales ~ promotion1',df).fit()
print(anova_lm(model))
#%%
model = ols('weekly_sales ~ promotion2',df).fit()
print(anova_lm(model))
#%%
model = ols('weekly_sales ~ promotion3',df).fit()
print(anova_lm(model))
#%%
model = ols('weekly_sales ~ promotion4',df).fit()
print(anova_lm(model))
#%%
model = ols('weekly_sales ~ promotion5',df).fit()
print(anova_lm(model))
#%%
model = ols('weekly_sales ~ temperature',df).fit()
print(anova_lm(model))
#%%
model = ols('weekly_sales ~ fuel_price',df).fit()
print(anova_lm(model))
#%%
model = ols('weekly_sales ~ unemployment',df).fit()
print(anova_lm(model))
#%%
df11 = df.copy()
df11 = df11.drop(['date'],axis=1)
#%%
df11 = df11.drop(['day','month','year','is_holiday_0','is_holiday_1'],axis=1)
#%%
df11 = df11.drop(['weekly_Sales','isholiday'],axis=1)
#%%
# 데이터 프레임 내 변수 박스플랏 그리기
f, ax = plt.subplots(figsize=(16, 14))
#ax.set_xscale("log")
ax = sns.boxplot(data = df11 , orient="h", palette="Set1")

ax.xaxis.grid(False)

plt.xlabel("Numeric values", fontsize = 10)
plt.ylabel("Feature names", fontsize = 10)
plt.title("Numeric Distribution of Features", fontsize = 15)
sns.despine(trim = True, left = True)
#%%
# promotion 시각화
sns.distplot(df_non_null['promotion1'],color="blue", label="promotion1")
sns.distplot(df_non_null['promotion2'],color="red", label="promotion2")
sns.distplot(df_non_null['promotion3'],color="green", label="promotion3")
sns.distplot(df_non_null['promotion4'],color="yellow", label="promotion4")
sns.distplot(df_non_null['promotion5'],color="gray", label="promotion5")
plt.legend(title="protion")
plt.title('결측치 0 채운 플랏, 분포 높이가 상당히 낮아짐')
plt.legend(prop={'size': 12}, title = 'Group')
plt.show()
#%%
pdf2 = pd.pivot_table(df,                # 피벗할 데이터프레임
                     index = 'store',    # 행 위치에 들어갈 열
                     columns = ['citytier','occupation','preferredpropertystar','monthlyincome'],    # 열 위치에 들어갈 열
                     values = 'promotion1',     # 데이터로 사용할 열
                     aggfunc = 'median')   # 데이터 집계함수
ppdf = pd.DataFrame(pdf2)   
print(ppdf.sort_values('promotion1',ascending=False))
#%%
plt.figure(figsize=(10,7))
sns.lineplot(data=df,x='date',y='promotion1',ci=None,hue='store',legend='full')
plt.legend([],[], frameon=False) # 범례 제거
plt.show()
#%%
plt.figure(figsize=(10,7))
sns.lineplot(data=df,x='date',y='promotion2',ci=None,hue='store',legend='full')
plt.legend([],[], frameon=False) # 범례 제거
plt.show()
#%%
plt.figure(figsize=(10,7))
sns.lineplot(data=df,x='date',y='promotion3',ci=None,hue='store',legend='full')
plt.legend([],[], frameon=False) # 범례 제거
plt.show()
print('일정기간 동안 매출이 급증함')
#%%
plt.figure(figsize=(10,7))
sns.lineplot(data=df,x='date',y='promotion4',ci=None,hue='store',legend='full')
plt.legend([],[], frameon=False) # 범례 제거
plt.show()
#%%
plt.figure(figsize=(10,7))
sns.lineplot(data=df,x='date',y='promotion5',ci=None,hue='store',legend='full')
plt.legend([],[], frameon=False) # 범례 제거
plt.show()
#%%
store1 = df[df['store']==1]
#%%
plt.figure(figsize=(10,7))
sns.lineplot(data=store1,x='date',y='promotion5',ci=None,hue='store',legend='full')
plt.legend([],[], frameon=False) # 범례 제거
plt.show()
#%%
sns.distplot(store1['promotion1'],color="gray", label="promotion1")
plt.title('Histogram & Density Plot by Groups')
plt.show()
#%%
pear_r,pear_pval = stats.kendalltau(df_non_null['promotion1'],df_non_null['weekly_sales'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')
#%%
pear_r,pear_pval = stats.kendalltau(df_non_null['temperature'],df_non_null['weekly_sales'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')
#%%
pear_r,pear_pval = stats.kendalltau(df_non_null['fuel_price'],df_non_null['weekly_sales'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')
#%%
pear_r,pear_pval = stats.kendalltau(df_non_null['unemployment'],df_non_null['weekly_sales'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')
#%%
df = df.drop(['is_holiday_1','promotion4','fuel_price'],axis=1)
#%%
train_data, val_data = train_test_split(df, test_size=0.2,random_state =  42) #20프로로 설정
train_data.reset_index(inplace=True) #전처리 과정에서 데이터가 뒤섞이지 않도록 인덱스를 초기화
val_data.reset_index(inplace=True)

#%%
train_data_X = train_data.drop(['id','weekly_sales','date','promotion2','promotion3'], axis = 1) #training 데이터에서 피쳐 추출
train_data_Y = train_data['weekly_sales'] #training 데이터에서 소비량 추출
#%%
val_data_X = val_data.drop(['id','weekly_sales','date','promotion2','promotion3'], axis = 1) #training 데이터에서 피쳐 추출
val_data_Y = val_data['weekly_sales'] #training 데이터에서 소비량 추출
#%%
dtrain = xgb.DMatrix(data=train_data_Y, label=train_data_Y)
dval = xgb.DMatrix(data=val_data_X, label=val_data_Y)
#dtest = xgb.DMatrix(data=test)
#%%
params = {'max_depth' : 3,
         'eta' : 0.1, 
         'objective' : 'binary:logistic',
         'eval_metric' : 'logloss',
         'early_stoppings' : 100 }
#%%
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_data_X,train_data_Y)
#%%
predd = tree_reg.predict(val_data_X)
predd
# In[104]:
from sklearn.metrics import mean_squared_error 
tree_mse = mean_squared_error(val_data_Y,predd)
tree_rmse = np.sqrt(tree_mse)
print(f'rmse:{tree_rmse}')
# In[105]:
train_data_X = df.drop(['id','weekly_sales','date','promotion2','promotion3'], axis = 1) #training 데이터에서 피쳐 추출
train_data_Y = df['weekly_sales'] #training 데이터에서 소비량 추출
# # 랜덤포레스트회귀
# In[110]:
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train_data_X,train_data_Y)
#%%
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
]
#%%
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv = 5,
                           return_train_score = True)
grid_search.fit(train_data_X,train_data_Y)
#%%
test = pd.read_csv('test.csv')
test.columns = test.columns.str.lower()
test['date'] = pd.to_datetime(test['date'],format='%d/%m/%Y')
test = test.astype({'store':'category'})
print(test.dtypes)
#%%
test['year'] = pd.DatetimeIndex(test['date']).year
test['month'] = pd.DatetimeIndex(test['date']).month
test['day'] = pd.DatetimeIndex(test['date']).day
print(df)
df = pd.get_dummies(df,columns=['is_holiday'])
print(df.head())
#%%
df_prom['isholiday'] = [1 if s == True else 0 for s in test['isholiday'].isnull()]
#%%
test['is_holiday_0'] = test['isholiday'].replace()
#%%
test= test.drop(labels='isholiday',axis=1)
print(test.info())
#%%
test= test.replace(np.nan, 0)
print(test.info())
#%%
test['promotion1'] = np.log1p(test['promotion1'])
test['promotion2'] = np.log1p(test['promotion2'])
test['promotion3'] = np.log1p(test['promotion3'])
test['promotion4'] = np.log1p(test['promotion4'])
test['promotion5'] = np.log1p(test['promotion5'])
#%%
test = test['promotion1'].fillna(0)
# In[105]:
test = test.drop(['id','date','promotion2','promotion3','promotion4'], axis = 1) #training 데이터에서 피쳐 추출

#%%
predd = grid_search.predict(test)
#%%