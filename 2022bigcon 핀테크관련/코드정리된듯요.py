#!/usr/bin/env python
# coding: utf-8

# In[1]:




# I
# In[58]:
train_gen_loan.reset_index(drop = False, inplace = True)
# In[59]:
train_gen_drop_na.reset_index(drop = False, inplace = True)
# In[60]:
# 추출한 데이터이기에 index가 맞지 않음. reset_index로 index 재부여.
train_gen_age.reset_index(drop = False, inplace = True)
# In[61]:
train_gen_enter.reset_index(drop = False, inplace = True)
# In[124]:
test_gen_loan.reset_index(drop = False, inplace = True)
# In[125]:
test_gen_drop_na.reset_index(drop = False, inplace = True)
# In[126]:
test_gen_age.reset_index(drop = False, inplace = True)
# In[127]:
test_gen_enter.reset_index(drop = False, inplace = True)
# In[62]:
# gender는 최빈값으로 채워줌.
train_gen_loan['gender'] = train_gen_loan['gender'].fillna(1)
# In[96]:
test_gen_loan['gender'] = test_gen_loan['gender'].fillna(1)
# In[64]:
# 단변량으로 유의한 변수 선택 후 그룹화. 그룹별 평균으로 결측치 대체.
for i in train_gen_loan.columns:
    model = ols('credit_score ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월', 'age'
train_gen_loan['credit_score'] = train_gen_loan['credit_score'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월', 'age'])['credit_score'].transform('mean'))
# In[66]:
for i in train_gen_loan.columns:
    model = ols('근속개월 ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'
train_gen_loan['근속개월'] = train_gen_loan['근속개월'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['근속개월'].transform('mean'))
# In[68]:
for i in train_gen_loan.columns:
    model = ols('age ~ train_gen_loan[i]', train_gen_loan).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt'
train_gen_loan['age'] = train_gen_loan['age'].fillna(train_gen_drop_na.groupby(['gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt'])['age'].transform('mean'))
# In[29]:
# loan_limit, loan_rate가 결측치인 데이터 셋임. 두 변수 제거.
train_gen_loan = train_gen_loan.drop(['loan_limit', 'loan_rate'], axis = 1)
# In[31]:
# 결측치 없는 거 확인.
train_gen_loan.isnull().sum()
# In[136]:
for i in train_gen_age.columns:
    model = ols('credit_score ~ train_gen_age[i]', train_gen_age).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n')
test_gen_age['credit_score'] = test_gen_age['credit_score'].fillna(train_gen_drop_na.groupby(['bank_id', 'loan_rate', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', '근속개월'])['credit_score'].transform('mean'))
# In[137]:
for i in train_gen_age.columns:
    model = ols('근속개월 ~ train_gen_age[i]', train_gen_age).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'loan_rate', 'employment_type', 'houseown_type', 'desired_amount',  'purpose', 'existing_loan_cnt'
test_gen_age['근속개월'] = test_gen_age['근속개월'].fillna(train_gen_drop_na.groupby(['loan_rate', 'employment_type', 'houseown_type', 'desired_amount',  'purpose', 'existing_loan_cnt'])['근속개월'].transform('mean'))
# In[140]:
# 결측치 없음을 확인.
test_gen_age.isnull().sum()
# In[141]:
# age, gender 변수 결측인 데이터 셋. 두 변수 제거.
test_gen_age = test_gen_age.drop(['gender', 'age'], axis = 1)
# test enter 결측치 처리
# In[143]:
for i in train_gen_enter.columns:
    model = ols('credit_score ~ train_gen_enter[i]', train_gen_enter).fit()
    print(f'독립변수 이름: {i}')
    print(anova_lm(model))
    print('============='*3)
    print(model.summary())
    print('\n') # 'bank_id', 'product_id', 'loan_rate', 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'
test_gen_enter['credit_score'] = test_gen_enter['credit_score'].fillna(train_gen_drop_na.groupby(['bank_id', 'product_id', 'loan_rate', 'gender', 'income_type', 'employment_type', 'houseown_type', 'purpose', 'existing_loan_cnt', 'age'])['credit_score'].transform('mean'))
# In[144]:
# 결측치 없음을 확인.
test_gen_enter.isnull().sum()
# In[145]:
# 근속개월 변수가 결측인 데이터 셋. 근속개월 변수 제거.
test_gen_enter = test_gen_enter.drop(['근속개월'], axis = 1)
# In[146]:
# 결측치가 없음을 확인.
test_gen_drop_na.isnull().sum()
#%%
# 스케일링 전 수치형과 문자형 데이터 분리.
train_gen_loan_num = train_gen_loan.copy()[['bank_id', 'product_id', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[193]:
train_gen_loan_ob = train_gen_loan.copy().drop(['bank_id', 'product_id', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'], axis = 1)
# In[195]:
test_gen_loan_num = test_gen_loan.copy()[['bank_id', 'product_id', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[196]:
test_gen_loan_ob = test_gen_loan.copy().drop(['bank_id', 'product_id', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'], axis = 1)
# In[197]:
train_gen_age_num = train_gen_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월']]
# In[198]:
train_gen_age_ob = train_gen_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월'], axis= 1)
# In[200]:
test_gen_age_num = test_gen_age.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월']]
# In[201]:
test_gen_age_ob = test_gen_age.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월'], axis= 1)
# In[202]:
train_gen_enter_num = train_gen_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', 'age']]
# In[203]:
train_gen_enter_ob = train_gen_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', 'age'], axis = 1)
# In[205]:
test_gen_enter_num = test_gen_enter.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', 'age']]
# In[206]:
test_gen_enter_ob = test_gen_enter.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', 'age'], axis = 1)
# In[207]:
train_gen_credit_num = train_gen_credit.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[208]:
train_gen_credit_ob = train_gen_credit.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'], axis= 1)
# In[209]:
test_gen_credit_num = test_gen_credit.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[210]:
test_gen_credit_ob = test_gen_credit.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'], axis= 1)
# In[211]:
train_gen_num = train_gen_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[212]:
train_gen_ob = train_gen_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
# In[213]:
test_gen_num = test_gen_drop_na.copy()[['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']]
# In[214]:
test_gen_ob = test_gen_drop_na.copy().drop(['bank_id', 'product_id', 'loan_limit', 'loan_rate', 
       'credit_score', 'yearly_income','desired_amount','existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age'],axis=1)
# In[216]:
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler()
# In[217]:
train_gen_loan_scaled = rbs.fit_transform(train_gen_loan_num) #fit시킨 데이터 적용
test_gen_loan_scaled = rbs.transform(test_gen_loan_num) #fit시킨 데이터 적용
# In[219]:
# np.array 형태를 DataFrame으로 변경.
train_gen_loan_scaled = pd.DataFrame(data = train_gen_loan_scaled )
test_gen_loan_scaled = pd.DataFrame(data = test_gen_loan_scaled)
# In[222]:
# 컬럼명 지정.
train_gen_loan_scaled.columns = ['bank_id', 'product_id', 'credit_score', 'yearly_income',
       'desired_amount', 'existing_loan_cnt', 'existing_loan_amt', '근속개월',
       'age']
# In[225]:
test_gen_loan_scaled.columns = ['bank_id', 'product_id', 'credit_score', 'yearly_income',
       'desired_amount', 'existing_loan_cnt', 'existing_loan_amt', '근속개월',
       'age']
# In[227]:
train_gen_age_scaled = rbs.fit_transform(train_gen_age_num) #fit시킨 데이터 적용
test_gen_age_scaled = rbs.transform(test_gen_age_num) #fit시킨 데이터 적용
# In[228]:
train_gen_age_scaled = pd.DataFrame(data = train_gen_age_scaled )
test_gen_age_scaled = pd.DataFrame(data = test_gen_age_scaled)
# In[230]:
train_gen_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', '근속개월']
# In[231]:
test_gen_age_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', '근속개월']
# In[232]:
train_gen_enter_scaled = rbs.fit_transform(train_gen_enter_num) #fit시킨 데이터 적용
test_gen_enter_scaled = rbs.transform(test_gen_enter_num) #fit시킨 데이터 적용
# In[233]:
train_gen_enter_scaled = pd.DataFrame(data = train_gen_enter_scaled )
test_gen_enter_scaled = pd.DataFrame(data = test_gen_enter_scaled)
# In[235]:
train_gen_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', 'age']
# In[236]:
test_gen_enter_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', 'age']
# In[237]:
train_gen_credit_scaled = rbs.fit_transform(train_gen_credit_num) #fit시킨 데이터 적용
test_gen_credit_scaled = rbs.transform(test_gen_credit_num) #fit시킨 데이터 적용
# In[238]:
train_gen_credit_scaled = pd.DataFrame(data = train_gen_credit_scaled )
test_gen_credit_scaled = pd.DataFrame(data = test_gen_credit_scaled)
# In[240]:
train_gen_credit_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'yearly_income',
       'desired_amount', 'existing_loan_cnt', 'existing_loan_amt', '근속개월',
       'age']
# In[241]:
test_gen_credit_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'yearly_income',
       'desired_amount', 'existing_loan_cnt', 'existing_loan_amt', '근속개월',
       'age']

#%%
train_gen_drop_na_scaled = rbs.fit_transform(train_gen_num) #fit시킨 데이터 적용
test_gen_drop_na_scaled = rbs.transform(test_gen_num) #fit시킨 데이터 적용
# In[245]:
train_gen_drop_na_scaled = pd.DataFrame(data = train_gen_drop_na_scaled )
test_gen_drop_na_scaled = pd.DataFrame(data = test_gen_drop_na_scaled)
# In[247]:
train_gen_drop_na_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
# In[248]:
test_gen_drop_na_scaled.columns = ['bank_id', 'product_id', 'loan_limit', 'loan_rate', 'credit_score',
       'yearly_income', 'desired_amount', 'existing_loan_cnt',
       'existing_loan_amt', '근속개월', 'age']
# In[250]:
# index가 맞지 않아 concat()을 사용할 수 없음. reset_index()로 재배치.
train_gen_loan_ob.reset_index(drop = False, inplace = True)
# In[251]:
train_gen_age_ob.reset_index(drop = False, inplace = True

# In[252]:
train_gen_enter_ob.reset_index(drop = False, inplace = True)
# In[253]:
train_gen_credit_ob.reset_index(drop = False, inplace = True)
# In[254]:
train_gen_ob.reset_index(drop = False, inplace = True)
# In[255]:
# concat() num과 ob를 합침.
train_gen_loan_sca = pd.concat([train_gen_loan_ob,train_gen_loan_scaled], axis=1)
# In[261]:
test_gen_loan_sca = pd.concat([test_gen_loan_ob,test_gen_loan_scaled], axis=1)
# In[273]:
# loan_limit, loan_rate가 결측치인 데이터. 두 변수 제거.
train_gen_loan_sca = train_gen_loan_sca.drop(['loan_limit', 'loan_rate'], axis = 1)
# In[258]:
train_gen_age_sca = pd.concat([train_gen_age_ob,train_gen_age_scaled], axis=1)
# In[264]:
test_gen_age_sca = pd.concat([test_gen_age_ob,test_gen_age_scaled], axis=1)
#%%
train_gen_enter_sca = pd.concat([train_gen_enter_ob,train_gen_enter_scaled], axis=1)
# In[266]:
test_gen_enter_sca = pd.concat([test_gen_enter_ob,test_gen_enter_scaled], axis=1)
# In[269]:
train_gen_credit_sca = pd.concat([train_gen_credit_ob,train_gen_credit_scaled], axis=1)
# In[270]:
test_gen_credit_sca = pd.concat([test_gen_credit_ob,test_gen_credit_scaled], axis=1)
# In[271]:
train_gen_drop_na_sca = pd.concat([train_gen_ob,train_gen_drop_na_scaled], axis=1)
# In[274]:
test_gen_drop_na_sca = pd.concat([test_gen_ob,test_gen_drop_na_scaled], axis=1)
# In[25]:
# credit_score가 결측인 데이트 셋. credit_score 변수 제거.
train_gen_credit = train_gen_credit.drop(['credit_score'], axis = 1)
# In[26]:
test_gen_credit = test_gen_credit.drop(['credit_score'], axis = 1)
# In[27]:
# 모델링 전 범주형 변수들 더미변수로 변환.
final_loan = pd.get_dummies(train_gen_loan, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[29]:
# is_applied가 모두 1임. test set도 1로 채워줌.
(final_loan.is_applied.value_counts())
test_gen_loan['is_applied'] = test_gen_loan['is_applied'].fillna(1)
# In[96]:
final_age = pd.get_dummies(train_gen_age, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[99]:
X_train = final_age.drop(['Unnamed: 0', 'level_0', 'index','is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
y_train = final_age['is_applied']
# In[101]:
# is_applied가 0과 1의 비율이 20:1 정도로 편향이 심함. 오버샘플링 실시.
smote = SMOTE(random_state=42)
# In[102]:
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
# In[103]:
# XGBClassifier의 성능이 가장 좋게 나옴.
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
# In[104]:
xgb_age.fit( X_train, y_train)
# In[91]:
# ROC곡선으로 시각화 및 AUC 확인.
from sklearn.metrics import roc_curve
# In[92]:
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
# In[111]:
# 사용하지 않는 데이터 제거.
test_gen_age = test_gen_age.drop(['Unnamed: 0', 'index'], axis = 1)
# In[112]:
dum_test_age = pd.get_dummies(test_gen_age,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[113]:
x_test = dum_test_age.drop(['is_applied','houseown_type_자가', 'income_type_PRIVATEBUSINESS'],axis=1)
# In[115]:
test_age_predict = xgb_age.predict(x_test)
# In[116]:
# is_applied 예측값 데이터프레임으로 변경.
a1= pd.DataFrame(test_age_predict)
# In[117]:
test_gen_age.is_applied = a1
# In[118]:
test_gen_age.to_csv('test_gen_age예측완료1_cp.csv',encoding = 'cp949')
# In[119]:
# 같은 방법으로 나머지 데이터 예측.
final_enter = pd.get_dummies(train_gen_enter, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[120]:
(final_enter.is_applied.value_counts())
# In[122]:
X_train = final_enter.drop(['Unnamed: 0', 'level_0', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
y_train = final_enter['is_applied']
# In[130]:
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
# In[131]:
xgb_enter.fit( X_train, y_train)
# In[132]:
dum_test_enter = pd.get_dummies(test_gen_enter,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[133]:
x_test = dum_test_enter.drop(['Unnamed: 0', 'index', 'is_applied','houseown_type_자가', 'houseown_type_배우자'],axis=1)
# In[134]:
X_train.columns
# In[135]:
x_test.columns
# In[136]:
test_enter_predict = xgb_enter.predict(x_test)
# In[137]:
a1= pd.DataFrame(test_enter_predict)
# In[138]:
test_gen_enter.is_applied = a1
# In[140]:
test_gen_enter.to_csv('test_gen_enter예측완료1_cp.csv',encoding = 'cp949')
# In[142]:
# credit 예측
final_credit = pd.get_dummies(train_gen_credit, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[143]:
(final_credit.is_applied.value_counts())
# In[145]:
X_train = final_credit.drop(['Unnamed: 0', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
y_train = final_credit['is_applied']
# In[146]:
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
# In[147]:
xgb_credit.fit( X_train, y_train)
# In[153]:
dum_test_credit = pd.get_dummies(test_gen_credit,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[154]:
x_test = dum_test_credit.drop(['Unnamed: 0', 'Unnamed: 0.1', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
# In[157]:
test_credit_predict = xgb_credit.predict(x_test)
# In[158]:
a1= pd.DataFrame(test_credit_predict)
# In[159]:
test_gen_credit.is_applied = a1
# In[160]:
test_gen_credit.isnull().sum()
# In[161]:
test_gen_credit.to_csv('test_gen_credit예측완료1_cp.csv',encoding = 'cp949')
# In[162]:
# drop_na 예측
final_drop_na = pd.get_dummies(train_gen_drop_na, columns = ['income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[163]:
(final_drop_na.is_applied.value_counts())
# In[164]:
X_train = final_drop_na.drop(['Unnamed: 0', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
y_train = final_drop_na['is_applied']
# In[165]:
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
# In[ ]:
xgb_drop_na.fit( X_train, y_train)
# In[ ]:
dum_test_drop_na = pd.get_dummies(test_gen_drop_na,columns = [ 'income_type',
       'employment_type', 'houseown_type', 'purpose'])
# In[ ]:
x_test = dum_test_drop_na.drop(['Unnamed: 0', 'index', 'is_applied','houseown_type_자가','houseown_type_배우자', 'income_type_PRIVATEBUSINESS'],axis=1)
# In[ ]:
X_train.columns
# In[ ]:
x_test.columns
# In[ ]:
test_drop_na_predict = xgb1.predict(x_test)
# In[ ]:
a1= pd.DataFrame(test_drop_na_predict) 
# In[ ]:
test_gen_drop_na.is_applied = a1
# In[ ]:
test_gen_drop_na.isnull().sum()
# In[ ]:
test_gen_drop_na.to_csv('test_gen_drop_na예측완료1_cp.csv',encoding = 'cp949')