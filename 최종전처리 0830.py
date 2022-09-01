# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:59:55 2022

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
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pycaret import classification
from sklearn.preprocessing import KBinsDiscretizer #이산화
from sklearn.preprocessing import PolynomialFeatures #이산화
#%%
# 플랏 한글
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#%%
os.chdir('C:\\Users\\215-05\\Downloads\\data')
print(os.getcwd())
#%%
# 데이터 불러오기 및 변수명 소문자 변경
tr = pd.read_csv('train.csv')
tr.columns = tr.columns.str.lower()
print(tr.info())
#%%
# 변수 결측치 탐색
tr.isnull().sum()
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
#%%
# 결측치 탐색
print('여행계획인원 이상치')
tr["numberofpersonvisiting"].value_counts()
#%%
print('numberoffollowups')
tr["numberoffollowups"].value_counts()
#%%
print('numberoftrips 이상치')
print(tr["numberoftrips"].value_counts())
print(tr["numberoftrips"].describe())
#%%
#durationofpitch 확인
sns.distplot(tr['monthlyincome'])
plt.title('monthlyincome 플랏')
plt.show()
print(tr["monthlyincome"].describe())
#%%
# monthlyincome 이상치 확인
print(f'upper이상치:{25558+(1.5*(25558-20390))}')
print(f'lower이상치:{20390-(1.5*(25558-20390))}')
#%%
kb = KBinsDiscretizer(n_bins=10, strategy='uniform')
tr['du_bin1'] = kb.fit(tr['durationofpitch'])
#  이때 `bin_edges_` 메소드를 이용하여 저장되어진 경계값을 확인할 수 있다.
print("bin edges :\n", kb.bin_edges_ )
#%%
object_columns = tr.columns[tr.dtypes == 'object']
print('object 칼럼은 다음과 같습니다 : ', list(object_columns))
#%%
train_enc = tr.copy()

# 모든 문자형 변수에 대해 encoder를 적용합니다.
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(tr[o_col])
    tr[o_col] = encoder.transform(tr[o_col])

# 결과를 확인합니다.
train_enc
#%%
# 이산화
bins=np.linspace(tr.durationofpitch.min(), tr.durationofpitch.max(),5)
tr['du_bin'] = np.digitize(tr['durationofpitch'], bins)
print(tr[tr['durationofpitch'].isnull()==True]['du_bin'])
print(tr['du_bin'].value_counts())
print(tr['durationofpitch'].describe())
#%%
#프레젠테이션 기간과 프레젠테이션 후속조치가 연관이 있을까?
X = tr['durationofpitch']
Y = tr['numberoffollowups']

import matplotlib.pyplot as plt
plt.scatter(X, Y, alpha=0.5)
plt.title('TARGET ~ BMI')
plt.xlabel('BMI')
plt.ylabel('TARGET')
plt.show()
np.corrcoef(Y,X)
#%%
colormap = plt.cm.PuBu
plt.figure(figsize=(10, 8))
plt.title("Person Correlation of Features", y = 1.05, size = 15)
sns.heatmap(a.corr(), linewidths = 0.1, vmax = 1.0,
           square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 16})
#%%
print(tr[tr['du_bin']==1]['durationofpitch'].describe())
print(tr[tr['du_bin']==2]['durationofpitch'].describe())
print(tr[tr['du_bin']==3]['durationofpitch'].describe())
print(tr[tr['du_bin']==4]['durationofpitch'].describe())
print(tr[tr['du_bin']==5]['durationofpitch'].describe())

#%%
a = tr[['durationofpitch','du_bin']]
