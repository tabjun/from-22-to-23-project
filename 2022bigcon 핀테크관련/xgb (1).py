#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[2]:


from xgboost import XGBClassifier
from xgboost import plot_importance ## Feature Importance를 불러오기 위함
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# In[3]:


import os
os.chdir('C:\\Users\\215-05\\Downloads')
os.getcwd()


# In[4]:


train = pd.read_csv('train_gen_sca.csv')


# In[5]:


train.columns


# In[6]:


final = pd.get_dummies(train,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])


# In[7]:


final.columns


# In[8]:


final.is_applied.value_counts()


# In[9]:


final_1 = final[final['is_applied']==1]
final_0 = final[final['is_applied']==0]


# In[10]:

final_0.sample(frac=(1/18)).shape


# In[11]:


final_0_s = final_0.sample(frac = (1/18))


# In[12]:


final_same = pd.concat([final_0_s,final_1],axis=0)


# In[13]:


x = final_same.drop(['Unnamed: 0', 'Unnamed: 0.1', 'is_applied','houseown_type_자가'],axis=1)
y = final_same['is_applied']
y = y.astype('int64')


# In[14]:


X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=1234)



#%%
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
#%%
xgb1.fit( X_train, y_train)

# In[17]:


xgb_pred = xgb1.predict(X_val)
#%%
print(classification_report(y_val, xgb_pred, target_names=['class 0', 'class 1']))

#%%
from sklearn.metrics import roc_curve
pred_positive_label = xgb1.predict_proba(X_val)[:,1]

fprs, tprs, thresholds = roc_curve(y_val, pred_positive_label)

 

precisions, recalls, thresholds = roc_curve(y_val, pred_positive_label)

 

plt.figure(figsize=(15, 5))

 

plt.plot([0,1], [0, 1], label='STR')

 

plt.plot(fprs, tprs, label='ROC')

 

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend()

plt.grid()

plt.show()

#%%
from sklearn.metrics import roc_auc_score
print(f'auc : {roc_auc_score(y_val,xgb_pred)}')
# In[19]:


from sklearn.model_selection import GridSearchCV

#%%
# 플랏 한글
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#%%
xgb1.fit( X_train, y_train)

# In[17]:


xgb_pred = xgb1.predict(X_val)
# In[21]:

shap.initjs()
explainer = shap.TreeExplainer(xgb1) # Tree model Shap Value 확인 객체 지정
shap_values = explainer.shap_values(X_train) # Shap Values 계산
shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :])

shap.summary_plot(shap_values, X_train)
# In[22]:


shap_interaction_values = explainer.shap_interaction_values(X_train)
shap.summary_plot(shap_interaction_values, X_train)


# In[23]:


xgb_param_grid={
    'n_estimators' : [100,200,300,400,500],
    'learning_rate' : [0.01,0.05,0.1,0.15],
    'max_depth' : [3,5,7,10,15],
    'gamma' : [0,1,2,3],
    'colsample_bytree' : [0.8,0.9],
    
}

#score종류는 acc,f1,f1_micro,f1_macro등 원하는걸로 설정)
#여기서 설정 파라미터의 갯수(총 4000개의 조합이므로 4000번의 학습이 돌아감)
xgb_grid=GridSearchCV(xgb, param_grid = xgb_param_grid, scoring="f1_macro", n_jobs=-1, verbose = 2)
# In[24]:
xgb_grid.fit(X_train,y_train)
# In[25]
final = xgb_grid.best_estimator_
# In[26]
final_pred = final.predict(X_val)
# In[27]:
print(classification_report(y_val, final_pred, target_names=['class 0', 'class 1']))
# In[28]:

shap.initjs()
explainer = shap.TreeExplainer(final) # Tree model Shap Value 확인 객체 지정
shap_values = explainer.shap_values(X_train) # Shap Values 계산
shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :])

shap.summary_plot(shap_values, X_train)
# In[ ]:


shap_interaction_values = explainer.shap_interaction_values(X_train)
shap.summary_plot(shap_interaction_values, X_train)


# In[ ]:


test = pd.read_csv('test_gen_sca.csv')

