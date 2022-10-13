#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


import os
os.chdir('C:\\Users\\222-04\\Downloads\\gennn')
os.getcwd()


# In[8]:


train_gen_sca=pd.read_csv('train_gen_drop_na_sca.csv',encoding='cp949')
test_gen_sca=pd.read_csv('test_gen_drop_na_sca.csv',encoding='cp949')


# In[9]:


train_gen_sca.columns


# In[10]:


test_gen_sca.columns


# In[11]:


train_gen_sca.drop(['Unnamed: 0', 'level_0', 'index'], axis=1, inplace=True)


# In[13]:


test_gen_sca.drop(['Unnamed: 0', 'index'], axis=1, inplace=True)


# ## Sampling

# #### 1) 더미변수 

# In[20]:


train_gen_sca.columns


# In[21]:


final=pd.get_dummies(train_gen_sca, columns=['income_type', 'employment_type', 'houseown_type', 'purpose'])


# In[22]:


final.columns


# #### 2) sampling

# In[23]:


final.is_applied.value_counts()


# In[26]:


import seaborn as sns


# In[28]:


sns.countplot(x="is_applied", data=final)
plt.title('is_applied')
plt.show()


# # 불균형 데이터 비율 추출을 위한 층화추출

# In[29]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=777)
for train_idx, test_idx in split.split(final, final["is_applied"]):
    tr = final.loc[train_idx]
    val = final.loc[test_idx]


# In[30]:


print(tr["is_applied"].value_counts() / len(tr))
val["is_applied"].value_counts() / len(val)


# In[31]:


from xgboost import XGBClassifier
from xgboost import plot_importance ## Feature Importance를 불러오기 위함
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import classification_report


# In[32]:


x_train=tr.drop(['is_applied','houseown_type_자가'], axis=1)
y_train=tr['is_applied']


# In[33]:


x_val=val.drop(['is_applied','houseown_type_자가'], axis=1)
y_val=val['is_applied']


# In[34]:


x_train.shape, x_val.shape


# - Grid Search 적용 X

# In[35]:


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
xgb1.fit( x_train, y_train)


# In[36]:


pred = xgb1.predict(x_val)


# In[37]:


print(classification_report(y_val, pred, target_names=['class 0', 'class 1']))


# In[38]:


from sklearn.metrics import roc_auc_score
print('roc_auc_score {}'.format(roc_auc_score(y_val, pred)))


# In[39]:


from sklearn.metrics import roc_curve

pred_positive_label = xgb1.predict_proba(x_val)[:,1]

fprs, tprs, thresholds = roc_curve(y_val, pred_positive_label)

print('샘플 추츨')
print()

thr_idx = np.arange(1, thresholds.shape[0], 6)
print('thr idx:', thr_idx)
print('thr thresholds value:', thresholds[thr_idx])
print('thr thresholds value:', fprs[thr_idx])
print('thr thresholds value:', tprs[thr_idx])


# In[40]:


pred_positive_label = xgb1.predict_proba(x_val)[:,1]
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


# ## test set 모델링

# In[41]:


test_gen_sca.isnull().sum()


# In[42]:


## 더미변수 생성
test=pd.get_dummies(test_gen_sca, columns=['income_type', 'employment_type', 'houseown_type', 'purpose'])


# In[43]:


test_x=test.drop(['is_applied','houseown_type_자가'], axis=1)


# In[48]:


xgb1_pred=xgb1.predict(test_x)
test_x.tail()


# In[49]:


a1 = pd.DataFrame(xgb1_pred)
a1.tail()


# In[50]:


a1.head()


# In[51]:


gen_drop_na = test_gen_sca.copy()
gen_drop_na['is_applied']=xgb1_pred
gen_drop_na.tail()


# In[52]:


gen_drop_na.to_csv('C:\\Users\\222-04\\Desktop\\gen_drop_na예측완_1014.csv')


# In[52]:


test.is_applied.value_counts()


# In[53]:


test.is_applied.value_counts()/len(test)


# In[ ]:




