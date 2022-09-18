# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:39:25 2022

@author: 215-05
"""


#%%
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
from pycaret.classification import *
from sklearn.preprocessing import RobustScaler
from impyute.imputation.cs import mice
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
#%%
# fe male -> female
tr['gender'].loc[tr['gender'] == 'Fe Male'] = 'Female'
print(tr['gender'].value_counts())
#%%
#범주형 값 처리위를 한 더미변수화
dum = pd.get_dummies(tr.copy())
print(dum.info())
#%%
# 결측치 다중 대치
no_null = pd.DataFrame(IterativeImputer(verbose=False).fit_transform(dum))
print(no_null.info())
#%%
#변수명 대체
no_null.columns = list(dum.columns)
#%%
o_x = no_null.copy().drop(['id','prodtaken'],axis=1)
o_y = no_null.copy()['prodtaken'] 
#%%
model = ols('o_y ~ o_x', no_null).fit()                                                                                
print(anova_lm(model))
print(model.summary())
#%%
print(no_null.info())
#%%
# 데이터 프레임 내 변수 박스플랏 그리기
f, ax = plt.subplots(figsize=(16, 14))
ax.set_xscale("log")
ax = sns.boxplot(data = no_null , orient="h", palette="Set1")

ax.xaxis.grid(False)

plt.xlabel("Numeric values", fontsize = 10)
plt.ylabel("Feature names", fontsize = 10)
plt.title("Numeric Distribution of Features", fontsize = 15)
sns.despine(trim = True, left = True)
# numberofpersonvisiting, numberoffollowups, numberoftripsm, montlyincome이상치 존재
#%%
exp_clf = setup(data = no_null, target = 'prodtaken',fold_shuffle = True,normalize=True,normalize_method='robust',
                session_id=1234,ignore_features=['id'])
#%%
#정확도기준 모델 형성 및 성능 비교
top5_model = compare_models(sort='Accuracy',fold=3,n_select = 5)
#%%
print(top5_model)
#%%
#f1스코어
top5_model_f1 = compare_models(fold=3, n_select = 5, sort='F1')
print(top5_model_f1)
# 정밀도
top5_model_prec = compare_models(fold=3, n_select = 5, sort='Precision')
print(top5_model_prec)
# 재현율
top5_model_recall = compare_models(fold=3, n_select = 5, sort='Recall')
print(top5_model_prec)






#%%
# 테스트 셋
test = pd.read_csv('test.csv')
test.columns = test.columns.str.lower()
print(test.info())
print(test.isnull().sum())
#%%
#범주형 값 처리위를 한 더미변수화
dum_test = pd.get_dummies(test.copy())
print(dum_test.info())
#%%
# 결측치 다중 대치
no_null_test = pd.DataFrame(IterativeImputer(verbose=False).fit_transform(dum_test))
print(no_null_test.info())
#%%
#변수명 대체
no_null_test.columns = list(dum_test.columns)
#%%
print(no_null_test.info())
#%%
# 모델 생성
nb = create_model('nb')
et = create_model('et')
gpc = create_model('gpc')
#%%
#모델 튜닝
tuned_nb = tune_model(nb,fold = 3 , optimize='Accuracy',choose_better=True)
tuned_et = tune_model(et,fold = 3 , optimize='Accuracy',choose_better=True)
tuned_gpc = tune_model(gpc,fold = 3, optimize='Accuracy',choose_better=True)
#%%
#엑스트라 트리
final_et = finalize_model(tuned_et)
pred_et = predict_model(final_et, data = no_null_test)
#%%
pred_et.to_csv('C:\\Users\\215-05\\Desktop\\여행여부 예측\\et_다중대치.csv',
               columns = ['Label'])
#%%
plot_model(tuned_et, plot = 'confusion_matrix')
plot_model(tuned_et, plot = 'auc')
interpret_model(tuned_et,plot = 'summary')
#%%
plot_model(tuned_nb, plot = 'confusion_matrix')
plot_model(tuned_nb, plot = 'auc')
interpret_model(tuned_nb,plot = 'summary')
#%%
plot_model(tuned_gpc, plot = 'confusion_matrix')
plot_model(tuned_gpc, plot = 'auc')
interpret_model(tuned_gpc,plot = 'summary')
