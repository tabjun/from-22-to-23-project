/* 3�� �� �ܺ��� */
/* �� 3�� ó�� */ 

/* import full */
filename a 'C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\��_3�� ����\����߰�_full_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_3_full
replace;
getnames=yes;
run;
proc print data= cow_3_full (obs=5);
run;

/* import train */
filename a 'C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\��_3�� ����\����߰�_train_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_3_train
replace;
getnames=yes;
run;
proc print data= cow_3_train (obs=5);
run;

/* import test */
filename a 'C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\��_3�� ����\����߰�_test_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_3_val
replace;
getnames=yes;
run;
proc print data= cow_3_val (obs=5);
run;

/* univariate logistic */
ods listing close;
ods pdf file='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\��_3�� ����\��_��_�ܺ���_���.pdf';

title '����';
proc logistic data=cow_3_train;
class gender (ref='N_M') /param = ref;
model target(event='1') = gender  / link = glogit;
run;

title '���ళ��';
proc logistic data=cow_3_train;
class target(ref='1')/param = ref;
model target(event='1') = sl_m  / link = glogit;
run;

title '��ü��';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_w  / link = glogit;
run;

title '���� ��ü�� ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_m_w  / link = glogit;
run;

title '���� ��ɴܸ��� ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_m_i  / link = glogit;
run;

title '���� ������ ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_f_m  / link = glogit;
run;

title '���� �ٳ����� ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_t_m  / link = glogit;
run;

title '���� ������';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_c  / link = glogit;
run;

title '������� ��ü�� ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = a_s_m_w  / link = glogit;
run;

title '������� ��ɴܸ��� ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = a_s_m_i  / link = glogit;
run;

title '������� ������ ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = a_s_f_m  / link = glogit;
run;

title '������� �ٳ����� ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') =a_s_t_m  / link = glogit;
run;

title '������� ������';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = a_s_c  / link = glogit;
run;

title '���ҹ����� ��ü�� ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = b_s_m_w  / link = glogit;
run;

title '���ҹ����� ��ɴܸ��� ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = b_s_m_i  / link = glogit;
run;

title '���ҹ����� ������ ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = b_s_f_m  / link = glogit;
run;

title '���ҹ����� �ٳ����� ���';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') =b_s_t_m  / link = glogit;
run;

title '���ҹ����� ������';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = b_s_c  / link = glogit;
run;

title '���� ���';
proc logistic data=cow_3_train;
class farm_level(ref='C') /param = ref;
model target(event='1') = farm_level  / link = glogit;
run;

ods pdf close;
ods listing;

/*3�� Multi. Logistic. Regression */

title '3�� �� �ٺ��� ������ƽ, forward selection';
proc logistic data=cow_3_train;
class farm_level(ref='C') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = forward slentry = 0.05;
run;

title '3�� �� �ٺ��� ������ƽ, backward selection';
proc logistic data=cow_3_train;
class farm_level(ref='C') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = backward slstay = 0.05;
run;

title '3�� �� �ٺ��� ������ƽ, stepwise selection';
proc logistic data=cow_3_train plots=roc;
class farm_level(ref='C') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = stepwise slentry = 0.05 slstay = 0.05;
run;

title '3�� �� �ٺ��� ������ƽ, ���� ������, ������, ������� ������ ��� ����';
proc logistic data=cow_3_train plots=roc;
class farm_level(ref='C') gender (ref='N_M') /param = ref;
model target(event='1') =gender s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_m_i a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit;
run;

/* ������跮 */
proc means data=cow_3_full n nmiss min max mean std maxdec=1;
class gender;
var s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_m_i a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c;
run;

proc means data=cow_3_train n nmiss min max mean std maxdec=1;
class gender;
var s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_m_i a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c;
run;


proc means data=cow_3_test n nmiss min max mean std maxdec=1;
class gender;
var s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_m_i a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c;
run;

/*3�� �� ���� ���� ���� */
/*train set�� confusion matirx auc,sesi, spec�̱� */
title1 'Uni&Multi, train���� validation�ٷ� ����, target:1{7,8,9}';
ods graphics on;
proc logistic data=cow_3_train;
class farm_level(ref='A') / param=ref;
model target(event='1') = s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ outroc=troc;
score data=cow_3_test out=valpred outroc=vroc;
roc; roccontrast;
run;

proc print data=valpred (obs=100);
run;


title1 'Uni&Multi';
proc logistic data=cow_3 plots(maxpoints=none)=roc;
class farm_level(ref='C') /param = ref;
model target(event='1') = s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ rocci;
output out= two p = phat;
store uni_multi;
run;

title1 'Uni&Multi';
proc logistic data=cow_3 plots(maxpoints=none)=roc;
class farm_level(ref='A') /param = ref;
model target(event='1') = s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ link = glogit rocci;
output out= two p = phat;
store uni_multi;
run;

/* train ������ ���� proba_���Ѵ��� phat�� ���, 0.5 cut_off�� ��� yhat�� ����ֱ�*/ 
data three;
set two;
if phat>0.5 then yhat=1;
else yhat=0;
run;

/* ȥ����� �׸��� */ 
proc freq data=three;
tables target*yhat/ senspec nocol norow nopercent;
run;

title1 'Variable Selection';
proc logistic data=cow_3 plots(maxpoints=none)=roc ;
class farm_level(ref='A') /param = ref;
model target(event='1') = sl_m s_w s_m_w s_m_i s_f_m s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ link = glogit rocci;
output out= two p = phat;
run;

/* train ������ ���� proba_���Ѵ��� phat�� ���, 0.5 cut_off�� ��� yhat�� ����ֱ�*/ 
data three;
set two;
if phat>0.5 then yhat=1;
else yhat=0;
run;

/* ȥ����� �׸��� */ 
proc freq data=three;
tables target*yhat/ senspec nocol norow nopercent measures;
run;

title1 'Uni&Multi';
proc logistic data=cow_3;
class farm_level(ref='C') /param = ref;
model target(event='1') = s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ link = glogit rocci;
output out= two p = phat;
store uni_multi;
run;

/* train ������ ���� proba_���Ѵ��� phat�� ���, 0.5 cut_off�� ��� yhat�� ����ֱ�*/ 
data three;
set two;
if phat>0.5 then yhat=1;
else yhat=0;
run;

/* ȥ����� �׸��� */ 
proc freq data=three;
tables target*yhat/ senspec nocol norow nopercent measures;
run;

/* validation ���� */
proc plm restore=uni_multi;
score data=cow_3_test out= val_3 / ilink;
run;

proc print data=val_3 (obs=5);
run;

/* 0.5 cut-off ���� */
data val_3_c;
set val_3;
if predicted > 0.5 then predict= 1;
else predict = 0;
run;

title1 '3�� multiple ���';
proc freq data=val_3_c order=data;
tables target*predict /senspec nocol norow nopercent;
run;


/* selection validation ���� */
proc plm restore=selection;
score data=cow_3_test out= val_3 / ilink;
run;

proc print data=val_3 (obs=10);
run;

/* 0.5 cut-off ���� */
data val_3_c;
set val_3;
if predicted > 0.5 then predict= 1;
else predict = 0;
run;

title1 '3�� selection ���';
proc freq data=val_3_c order=data;
tables target*predict /senspec nocol norow nopercent;
run;
