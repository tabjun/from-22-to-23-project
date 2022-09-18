#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("C:\\Users\\user\\Desktop\\경진대회\\데이콘\\소비자 데이터 기반 소비량 예측\\마켓 캠페인")
print(os.getcwd())


# In[2]:


import numpy as np
import statsmodels as stats
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import scipy.stats as stats
import seaborn as sns


# In[3]:


a = pd.read_csv("train123.csv")
a.head()


# In[4]:


print(a.columns)


# In[5]:


a.columns = ['id','birth','edu','maritial','income','kid','teen','dt_cus','recency','sales',
             'webpur','catalog','store','webvisit','cmp3','cmp4','cmp5','cmp1','cmp2',
             'complain','response','target']


# In[6]:


print(f'변경한 변수명:{a.columns}')


# # 결측치가 없음

# In[7]:


print(a.info()) #전체 데이터형 파악


# In[8]:


print(a.isnull().sum())


# # 범주값인지 확인

# In[9]:


print(a['edu'].unique())
print(a['cmp1'].unique())
print(a['maritial'].unique())
print(a['complain'].unique())
print(a['response'].unique())


# In[10]:


a_copy = a.copy()


# In[11]:


a_copy.describe()


# In[12]:


a['dt_cus']


# In[13]:


a_copy['dt_cus'] = pd.to_datetime(a['dt_cus'])


# In[14]:


a_copy.describe(include = ['object','category'])


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp

# Probability density of the Poisson distribution
def pois_dist(n, lamb):
    pd = (lamb ** n) * exp(-lamb) / factorial(n)
    return pd


x = np.arange(40)
pd1 = np.array([pois_dist(n, 10) for n in range(40)])
plt.ylim(0, 0.15)
plt.text(33.5, 0.14, 'lamb = 10')
plt.plot(x, pd1, color='lightcoral')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp


# In[16]:


import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# In[17]:


sns.distplot(a_copy['sales'])
plt.title('할인 품목 구매수')
plt.show()


# In[18]:


sns.distplot(a_copy['target'])
plt.title('소비량')
plt.show()


# In[19]:


sns.distplot(a_copy['webpur'])
plt.title('웹 구매 수')
plt.show()


# In[20]:


sns.distplot(a_copy['catalog'])
plt.title('카탈로그 통한 구매 수')
plt.show()


# In[21]:


sns.distplot(a_copy['webvisit'])
plt.title('웹 방문 수')
plt.show()


# In[22]:


sns.distplot(a_copy['store'])
plt.title('매장 방문 구매 수')
plt.show()


# # 포아송분포 
# - 등분산성을 만족하면 안됨 검정 해줌
# - 정규성 만족하는지 안하는지 따라
# - 정규성 만족: Bartlett - Test
# - 정규성 만족하지 못하면: Levene - Test

# In[23]:


# 정규성 검정해주기
from scipy.stats import shapiro
print('''
귀무가설 H0: 데이터셋이 정규분포를 따른다.
대립가설 H1: 데이터셋이 정규분포를 따르지 않는다.
''')

통계량,pval=shapiro(a['income'])
print(f'income 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['birth'])
print(f'birth 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['recency'])
print(f'recency 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['kid'])
print(f'kid 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['teen'])
print(f'teen 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['sales'])
print(f'sales 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['webpur'])
print(f'webpur 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['catalog'])
print(f'catalog 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['store'])
print(f'store 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['webvisit'])
print(f'webvisit 레빈 계수:{통계량},유의확률:{pval})')
print('========'*10)
통계량,pval=shapiro(a['target'])
print(f'target 레빈 계수:{통계량},유의확률:{pval})')
print('\n')
print('모두 정규성을 만족하지 못함')


# In[24]:


#%%
# 등분산검정
print('''
      H_{0} : 등분산성을 만족한다
      H_{1} : 등분산성을 만족하지 못한다
      ''')

from scipy.stats import levene
sta,pval=levene(a['income'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['birth'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['income'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['recency'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['kid'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['teen'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['sales'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['webpur'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['catalog'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['store'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')
print('========'*10)
sta,pval=levene(a['webvisit'],a['target'])
print(f'레빈 계수:{sta},유의확률:{pval})')


# - 포아송분포 사용 가능

# # income의 이상치

# In[25]:


print(a['income'].describe())


# In[26]:


print(f'low이상치:{35768.5-(1.5*(68325-35768.5))}')
print(f'high이상치:{68325+(1.5*(68325-35768.5))}')


# In[27]:


a_copy.columns


# In[28]:


plt.scatter(a_copy['income'],a_copy['sales'],alpha=0.2)
plt.xlabel('소득')
plt.ylabel('할인구매')
plt.show()


# In[29]:


plt.scatter(a['income'],a['target'],alpha=0.2)
plt.show()


# In[30]:


plt.scatter(a_copy['catalog'],a_copy['target'],alpha=0.2)
plt.show()


# In[31]:


plt.scatter(a_copy['birth'],a_copy['income'],alpha=0.2)
plt.show()


# ### 카탈로그는 상품을 구매할 것이 예상되는 손님에게 구입상 참고가 될 만한 사항을 나타내 보이는 것. (특징,가격,사진 등)
# ### 소비량과 카탈로그의 구매수를 상관계수가 높은 것을 볼 수 있는데 참고가 될 만한 것들이 있을수록 더 많이 구매하는 것으로 생각할 수 있다. 
# 
# ## 소비량과 매장에서 직접 구매한 수도 상관계수가 높은데 매장에서 직접 제품을 보는 것은 카탈로그가 제공되는 것이기 때문에 카탈로그를 만들어 주는게 좋다.

# - 피어슨은 연속형 연속형에 정규성을 만족하는 변수들에 대하여 구해주는 것
# - 앞에 과정에서 정규성을 검정했을 때 

# - 결국 정규성을 만족하지 않아 비모수적 방법인 스피어만 상관계수로 분석함

# In[32]:


pear_r,pear_pval = stats.spearmanr(a_copy['birth'],a_copy['income'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# - 이상치 소득을 가진 사람들은 캠페인 참여도 적고 소비량도 작아서 별로 필요없는 애들이라 판단함

# In[33]:


print(a_copy[a_copy['income']>=117159.75][['cmp1','cmp2','cmp3','cmp4','cmp5','response','target']])


# In[34]:


pear_r,pear_pval = stats.kendalltau(a_copy['income'],a_copy['sales'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# - 유의확률은 0.000으로 귀무가설 기각 상관성이 있으며 상관계수는 -0.136으로 음의 상관계수가 존재한다.

# In[136]:


pear_r,pear_pval = stats.kendalltau(a_copy['income'],a_copy['target'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# - 유의확률은 0.000으로 귀무가설 기각 상관성이 있으며 상관계수는 0.78으로 양의 상관계수가 존재한다

# In[137]:


pear_r,pear_pval = stats.kendalltau(a_copy['catalog'],a_copy['target'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# - 유의확률은 0.000으로 귀무가설 기각 상관성이 있으며 상관계수는 0.80으로 양의 상관계수가 존재한다

# In[138]:


pear_r,pear_pval = stats.kendalltau(a_copy['store'],a_copy['target'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# - 유의확률은 0.000으로 귀무가설 기각 상관성이 있으며 상관계수는 0.68으로 양의 상관계수가 존재한다

# In[139]:


pear_r,pear_pval = stats.kendalltau(a_copy['webvisit'],a_copy['webpur'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# In[140]:


pear_r,pear_pval = stats.kendalltau(a_copy['webvisit'],a_copy['income'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# In[141]:


pear_r,pear_pval = stats.kendalltau(a_copy['birth'],a_copy['income'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# - 유의확률은 0.004으로 귀무가설 기각 상관성이 있으며 상관계수는 -0.08으로 음의 상관계수가 존재한다

# - 웹 방문 상관계수 넣고 이유 설명해주기

# In[37]:


a_del = a_copy.copy()
a_del = a_del[a_del['income']<117159.75]
a_del=a_del.drop(['webvisit'],axis=1)
a_del.shape


# - 이상치 3개 제거

# In[38]:


#%%
conditionlist = [
    (a_del['birth'] >= 1890) & (a_del['birth'] < 1900),
    (a_del['birth'] >= 1900) & (a_del['birth'] < 1910),
    (a_del['birth'] >= 1910) & (a_del['birth'] < 1920),
    (a_del['birth'] >= 1920) & (a_del['birth'] < 1930),
    (a_del['birth'] >= 1930) & (a_del['birth'] < 1940),
    (a_del['birth'] >= 1940) & (a_del['birth'] < 1950),
    (a_del['birth'] >= 1950) & (a_del['birth'] < 1960),
    (a_del['birth'] >= 1960) & (a_del['birth'] < 1970),
    (a_del['birth'] >= 1970) & (a_del['birth'] < 1980),
    (a_del['birth'] >= 1980) & (a_del['birth'] < 1990),
    (a_del['birth'] >= 1990) & (a_del['birth'] < 2000)]
choicelist = ['1890년대 출생', '1900년대 출생','1910년대 출생','1920년대 출생','1930년대 출생',
              '1940년대 출생','1950년대 출생','1960년대 출생','1970년대 출생','1980년대 출생',
              '1990년대 출생']
a_del['birth_level'] = np.select(conditionlist, choicelist,default=0)
print(a_del.head())


# In[144]:


pear_r,pear_pval = stats.kendalltau(a_del['birth'],a_del['income'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# In[145]:


pear_r,pear_pval = stats.kendalltau(a_del['income'],a_del['target'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# In[146]:


pear_r,pear_pval = stats.kendalltau(a_del['catalog'],a_del['target'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# In[147]:


pear_r,pear_pval = stats.kendalltau(a_del['income'],a_del['sales'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# - 폰트 한글가능하게

# In[39]:


from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# In[149]:


# 많음, 이산화
plt.figure(figsize=(12,5))
sns.countplot(x='birth_level',data=a_del)
plt.xticks(rotation=90);plt.title('Year_Birth distribution')
plt.title('나이 이산화 박스플랏')
plt.show()
print('1900,1890,1940,1990 거의 없음 묶어주는게')


# In[40]:


print(a_del['birth_level'].value_counts())


# In[41]:


a_del.loc[a_del.birth_level == '1890년대 출생', 'birth_level'] = 'others'
a_del.loc[a_del.birth_level == '1900년대 출생', 'birth_level']= 'others'
a_del.loc[a_del.birth_level == '1940년대 출생', 'birth_level'] = 'others'
a_del.loc[a_del.birth_level == '1990년대 출생', 'birth_level'] = 'others'
a_del['birth_level']


# In[42]:


a_del['birth_level'].unique()


# In[43]:


# 많음, 이산화

sns.countplot(x='birth_level',data=a_del)
plt.xticks(rotation=90);plt.title('Year_Birth distribution')
plt.title('나이 이산화 박스플랏')
plt.show()


# In[44]:


pear_r,pear_pval = stats.kendalltau(a_del['income'],a_del['target'])
print(f'상관계수:{pear_r},유의확률:{pear_pval}')


# In[45]:


a_del.loc[a_del['dt_cus'] <= '2012-05-31', 'dt_group'] = '2012_half1'
a_del.loc[(a_del['dt_cus'] > '2012-05-31') & (a_del['dt_cus'] <= '2012-12-31'), 'dt_group'] = '2012_half2'
a_del.loc[(a_del['dt_cus'] > '2012-12-31') & (a_del['dt_cus'] <= '2013-05-31'), 'dt_group'] = '2013_half1'
a_del.loc[(a_del['dt_cus'] > '2013-05-31') & (a_del['dt_cus'] <= '2013-12-31'), 'dt_group'] = '2013_half2'
a_del.loc[(a_del['dt_cus'] > '2013-12-31') & (a_del['dt_cus'] <= '2014-5-31'), 'dt_group'] = '2014_half1'
a_del.loc[(a_del['dt_cus'] > '2014-05-31') & (a_del['dt_cus'] <= '2014-12-31'), 'dt_group'] = '2014_half2'


# In[46]:


a_del['dt_group'].value_counts()


# In[47]:


dt_cmp = a_del[['dt_group', 'cmp1', 'cmp2', 'cmp3','cmp4','cmp5','response','target' ]]


# In[48]:


dt_cmp['total_cmp'] = dt_cmp['cmp1'] + dt_cmp['cmp2'] + dt_cmp['cmp3'] + dt_cmp['cmp4'] + dt_cmp['cmp5']  + dt_cmp['response']


# In[49]:


pi_cmp = dt_cmp.groupby('dt_group').sum()
pi_cmp.sort_values(by = ['target'],ascending = False)


# - 전체적으로 캠페인 참여가 높은 대로 소비량 또한 높은 것을 볼 수 있으며 cmp2는 낮은 참여율을 나타내고 있으므로
# - 좋은 사례의 캠페인이라고는 볼 수 없다 그래서 뺀다

# In[50]:


sns.stripplot(y = a_del['sales'],x = a_del['kid'])
plt.title('아이와 할인품목 구매 수')
plt.show()


# In[51]:


sns.stripplot(y = a_del['sales'],x = a_del['teen'])
plt.title('아이와 할인품목 구매 수')
plt.show()


# In[52]:


sns.stripplot(y = a_del['webpur'],x = a_del['teen'])
plt.title('청소년과 웹 구매 수')
plt.show()


# In[53]:


# 육아 중이면 바쁘니까 웹 구매횟수가 많지 않을까?
sns.stripplot(y = a_del['webpur'],x = a_del['kid'])
plt.title('아이와 웹 구매 수')
plt.show()


# In[54]:


# 뭔가 전체적으로 분포도 비슷하고 0 1 2 값도 똑같고 그냥 보유자녀로 묶자 괜히 변수 갯수 늘리지 말고
a_del['parch'] = a_del['kid'] + a_del['teen']
print(a_del['parch'].head())


# In[55]:


# In[84]:
print(pd.pivot_table(a_del,
    index = ['kid'],
    values = ['target'],
    aggfunc = ['count']))


# In[56]:


# In[84]:
print(pd.pivot_table(a_del,
    index = ['teen'],
    values = ['target'],
    aggfunc = ['count']))


# In[57]:


plt.bar(a_del['teen'].unique(),a_del['teen'].value_counts())
plt.xlabel('청소년')
plt.ylabel('빈도')
plt.title('청소년 수에 따른 빈도')
plt.show()


# In[58]:


plt.bar(a_del['kid'].unique(),a_del['kid'].value_counts())
plt.xlabel('아이')
plt.ylabel('빈도')
plt.title('아이 수에 따른 빈도')
plt.show()


# In[59]:


sns.stripplot(y = a_del['webpur'],x = a_del['parch'])
plt.title('청소년과 웹 구매 수')
plt.show()


# In[60]:


plt.bar(a_del['parch'].unique(),a_del['parch'].value_counts())
plt.xlabel('자녀')
plt.ylabel('빈도')
plt.title('자녀 수에 따른 빈도')
plt.show()


# In[61]:


# In[84]:
print(pd.pivot_table(a_del,
    index = ['parch'],
    values = ['target'],
    aggfunc = ['count']))


# - 전체적으로 아이나 청소년 수가 적을수록 소비량이 많음
# - 그리고 아이나 청소년 수의 분포가 비슷해서 그냥 합쳐서 파생변수 만들어서 분석 진행해도 될듯

# In[62]:


# 결혼상태 각 개수 파악
print(a_del['maritial'].value_counts()) 


# In[63]:


# 결혼상태 각 개수 파악
print(a_del['maritial'].value_counts()) 


# In[64]:


a_del.columns


# In[85]:


a_del.shape


# In[66]:


a_del.dtypes


# # 포아송 회귀분석
# - 지수분포족 형태를 하고 있으며 2년이내에 소비한 횟수이므로 단위시간당 시행횟수인 포아송분포를 이용한 분석방법이 필요하다고 판단
# - 포아송분포를 사용하기 위한 검정을 끝냈음
# - 회귀식 유의성 검정하고 식 세우고
# - 이런 식이 세워졌습니다 하고
# - 해석 좀  해주고 

# # 모델링 
# - 모델링 좀 해주고 
# - 결과 및 해석 
# - 모델 개선

# In[83]:


a_del_dum = pd.get_dummies(a_del[['cmp1','cmp3','cmp4','cmp5','response','edu','maritial','birth_level','dt_group','complain']],drop_first = True)
aaa = pd.concat([a_del,a_del_dum],axis = 1)
aaa.head(10)


# In[87]:


aaa = aaa.drop(['id','birth','kid','teen','dt_cus','cmp2','birth_level','dt_group','maritial','edu','cmp1','cmp3','cmp4','cmp5','response','complain'], axis = 1 )


# In[88]:


aaa.info()


# In[89]:


aaa.shape


# In[90]:


from sklearn.model_selection import train_test_split
target = aaa['target']


# # 회귀트리

# In[98]:


#필요없는 id열 삭제
train_data, val_data = train_test_split(aaa, test_size=0.2,random_state =  42) #20프로로 설정
train_data.reset_index(inplace=True) #전처리 과정에서 데이터가 뒤섞이지 않도록 인덱스를 초기화
val_data.reset_index(inplace=True)


# In[99]:


print('학습시킬 train 셋 : ', train_data.shape)
print('검증할 val 셋 : ', val_data.shape)


# In[100]:


train_data_X = train_data.drop(['target', 'index'], axis = 1) #training 데이터에서 피쳐 추출
train_data_Y = train_data.target #training 데이터에서 소비량 추출


# In[101]:


val_data_X = val_data.drop(['target', 'index'], axis = 1) #training 데이터에서 피쳐 추출
val_data_Y = val_data.target #training 데이터에서 소비량 추출


# In[102]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_data_X,train_data_Y)


# In[103]:


predd = tree_reg.predict(val_data_X)
predd


# In[104]:


from sklearn.metrics import mean_squared_error 
tree_mse = mean_squared_error(val_data_Y,predd)
tree_rmse = np.sqrt(tree_mse)
print(f'rmse:{tree_rmse}')


# In[105]:


from sklearn.metrics import mean_absolute_error
tree_mae = mean_absolute_error(val_data_Y,predd)
print(f'mae:{tree_mae}')


# In[106]:


a_del['target'].describe()


# In[107]:


tree_mse


# In[108]:


train_data_X = aaa.drop(['target'], axis = 1) #training 데이터에서 피쳐 추출
train_data_Y = aaa['target'] #training 데이터에서 소비량 추출


# # 랜덤포레스트회귀

# In[110]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train_data_X,train_data_Y)


# - 매개변수 조합

# In[111]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
]


# In[112]:


forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv = 5,
                          scoring='mean_absolute_error',
                          return_train_score = True)
grid_search.fit(train_data_X,train_data_Y)


# In[113]:


grid_search.best_params_


# In[114]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[115]:


def NMAE(true, pred):
    score = np.mean(np.abs(true - pred) / true)
    return score


# In[116]:


from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score


# In[117]:


test_1 = pd.read_csv("test123.csv")
test_1.head()


# In[118]:


test_1.info()


# In[119]:


test_1.columns = ['id','birth','edu','maritial','income','kid','teen','dt_cus','recency','sales',
             'webpur','catalog','store','webvisit','cmp3','cmp4','cmp5','cmp1','cmp2',
             'complain','response']


# In[120]:


tt = test_1.copy()
#tt = tt[tt['income']<117159.75]
tt=tt.drop(['webvisit'],axis=1)
tt.shape


# In[121]:


tt['dt_cus'] = pd.to_datetime(tt['dt_cus'])


# In[122]:


conditionlist = [
    (tt['birth'] >= 1890) & (tt['birth'] < 1900),
    (tt['birth'] >= 1900) & (tt['birth'] < 1910),
    (tt['birth'] >= 1910) & (tt['birth'] < 1920),
    (tt['birth'] >= 1920) & (tt['birth'] < 1930),
    (tt['birth'] >= 1930) & (tt['birth'] < 1940),
    (tt['birth'] >= 1940) & (tt['birth'] < 1950),
    (tt['birth'] >= 1950) & (tt['birth'] < 1960),
    (tt['birth'] >= 1960) & (tt['birth'] < 1970),
    (tt['birth'] >= 1970) & (tt['birth'] < 1980),
    (tt['birth'] >= 1980) & (tt['birth'] < 1990),
    (tt['birth'] >= 1990) & (tt['birth'] < 2000)]
choicelist = ['1890년대 출생', '1900년대 출생','1910년대 출생','1920년대 출생','1930년대 출생',
              '1940년대 출생','1950년대 출생','1960년대 출생','1970년대 출생','1980년대 출생',
              '1990년대 출생']
tt['birth_level'] = np.select(conditionlist, choicelist,default=0)
print(tt.head())


# In[123]:


tt.loc[tt.birth_level == '1890년대 출생', 'birth_level'] = 'others'
tt.loc[tt.birth_level == '1900년대 출생', 'birth_level']= 'others'
tt.loc[tt.birth_level == '1940년대 출생', 'birth_level'] = 'others'
tt.loc[tt.birth_level == '1990년대 출생', 'birth_level'] = 'others'
tt['birth_level']


# In[124]:


tt['dt_cus']


# In[125]:


tt.loc[tt['dt_cus'] <= '2012-05-31', 'dt_group'] = '2012_half1'
tt.loc[(tt['dt_cus'] > '2012-05-31') & (tt['dt_cus'] <= '2012-12-31'), 'dt_group'] = '2012_half2'
tt.loc[(tt['dt_cus'] > '2012-12-31') & (tt['dt_cus'] <= '2013-05-31'), 'dt_group'] = '2013_half1'
tt.loc[(tt['dt_cus'] > '2013-05-31') & (tt['dt_cus'] <= '2013-12-31'), 'dt_group'] = '2013_half2'
tt.loc[(tt['dt_cus'] > '2013-12-31') & (tt['dt_cus'] <= '2014-5-31'), 'dt_group'] = '2014_half1'
tt.loc[(tt['dt_cus'] > '2014-05-31') & (tt['dt_cus'] <= '2014-12-31'), 'dt_group'] = '2014_half2'


# In[126]:


tt['parch'] = tt['kid'] + tt['teen']
print(tt['parch'].head())


# In[127]:


a_del_dum = pd.get_dummies(a_del[['cmp1','cmp3','cmp4','cmp5','response','edu','maritial','birth_level','dt_group','complain']],drop_first = True)
aaa = pd.concat([a_del,a_del_dum],axis = 1)
aaa.head(10)


# In[128]:


tt_dum = pd.get_dummies(tt[['cmp1','cmp3','cmp4','cmp5','response','edu','maritial','birth_level','dt_group','complain']],drop_first = True)
ttt = pd.concat([tt,tt_dum],axis = 1)
ttt.head(10)


# In[129]:


ttt.shape


# In[130]:


ttt = ttt.drop(['id','birth','kid','teen','dt_cus','cmp2','birth_level','dt_group','maritial','edu','cmp1','cmp3','cmp4','cmp5','response','complain'], axis = 1 )


# In[131]:


ttt.shape


# In[132]:


forest_reg.fit(train_data_X,train_data_Y)


# In[137]:


abcd = pd.DataFrame(forest_reg.predict(ttt))


# In[138]:


abcd.to_csv('C:\\Users\\user\\Desktop\\경진대회\\마켓 캠페인\\예측값.csv',sep = ',')


# In[ ]:


tree_mae = mean_absolute_error(,predd)
print(f'mae:{tree_mae}')

