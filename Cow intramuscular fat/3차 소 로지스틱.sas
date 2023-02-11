/* �� 5�� ó�� */ 
filename a 'C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\��_5�� ����\����߰�_train_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_5
replace;
getnames=yes ;
run;
proc print data= cow_5 (obs=5);
run;

/* univariate logistic */
ods listing close;
ods pdf file='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\��_5�� ����\5��_��_�ܺ���_���.pdf';

title '����';
proc logistic data=cow_5 plots=roc;
class gender (ref='N_M') /param = ref;
model target(event='1') = gender  / link = glogit;
run;

title '���ళ��';
proc logistic data=cow_5;
class target(ref='1')/param = ref;
model target(event='1') = sl_m  / link = glogit;
run;

title '��ü��';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_w  / link = glogit;
run;

title '���� ��ü�� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_m_w  / link = glogit;
run;

title '���� ��ɴܸ��� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_m_i  / link = glogit;
run;

title '���� ������ ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_f_m  / link = glogit;
run;

title '���� �ٳ����� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_t_m  / link = glogit;
run;

title '���� ������';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_c  / link = glogit;
run;

title '������� ��ü�� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = a_s_m_w  / link = glogit;
run;

title '������� ��ɴܸ��� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = a_s_m_i  / link = glogit;
run;

title '������� ������ ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = a_s_f_m  / link = glogit;
run;

title '������� �ٳ����� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') =a_s_t_m  / link = glogit;
run;

title '������� ������';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = a_s_c  / link = glogit;
run;

title '���ҹ����� ��ü�� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_m_w  / link = glogit;
run;

title '���ҹ����� ��ɴܸ��� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_m_i  / link = glogit;
run;

title '���ҹ����� ������ ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_f_m  / link = glogit;
run;

title '���ҹ����� �ٳ����� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_t_m  / link = glogit;
run;

title '���ҹ����� ������';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_c  / link = glogit;
run;

title '���������� ��ü�� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_m_w  / link = glogit;
run;

title '���������� ��ɴܸ��� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_m_i  / link = glogit;
run;

title '���������� ������ ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_f_m  / link = glogit;
run;

title '���������� �ٳ����� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_t_m  / link = glogit;
run;

title '���������� ������';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_c  / link = glogit;
run;

title '�ܰ������� ��ü�� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_m_w  / link = glogit;
run;

title '�ܰ������� ��ɴܸ��� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_m_i  / link = glogit;
run;

title '�ܰ������� ������ ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_f_m  / link = glogit;
run;

title '�ܰ������� �ٳ����� ���';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_t_m  / link = glogit;
run;

title '�ܰ������� ������';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_c  / link = glogit;
run;

title '���� ���';
proc logistic data=cow_5;
class farm_level(ref='C') /param = ref;
model target(event='1') = farm_level  / link = glogit;
run;

ods pdf close;
ods listing;
/* 3�� �� �ܺ��� */
/* �� 3�� ó�� */ 
filename a 'C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\��_3�� ����\����߰�_train_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_3
replace;
getnames=yes;
run;
proc print data= cow_3 (obs=10);
run;

/* univariate logistic */
ods listing close;
ods pdf file='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\��_3�� ����\��_��_�ܺ���_���.pdf';

title '����';
proc logistic data=cow_3;
class gender (ref='N_M') /param = ref;
model target(event='1') = gender  / link = glogit;
run;

title '���ళ��';
proc logistic data=cow_3;
class target(ref='1')/param = ref;
model target(event='1') = sl_m  / link = glogit;
run;

title '��ü��';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_w  / link = glogit;
run;

title '���� ��ü�� ���';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_m_w  / link = glogit;
run;

title '���� ��ɴܸ��� ���';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_m_i  / link = glogit;
run;

title '���� ������ ���';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_f_m  / link = glogit;
run;

title '���� �ٳ����� ���';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_t_m  / link = glogit;
run;

title '���� ������';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_c  / link = glogit;
run;

title '������� ��ü�� ���';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = a_s_m_w  / link = glogit;
run;

title '������� ��ɴܸ��� ���';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = a_s_m_i  / link = glogit;
run;

title '������� ������ ���';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = a_s_f_m  / link = glogit;
run;

title '������� �ٳ����� ���';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') =a_s_t_m  / link = glogit;
run;

title '������� ������';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = a_s_c  / link = glogit;
run;

title '���� ���';
proc logistic data=cow_3;
class farm_level(ref='C') /param = ref;
model target(event='1') = farm_level  / link = glogit;
run;

ods pdf close;
ods listing;

/*3�� Multi. Logistic. Regression */

title '3�� �� �ٺ��� ������ƽ, forward selection';
proc logistic data=cow_3 plots=roc;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = forward slentry = 0.05;
run;

title '3�� �� �ٺ��� ������ƽ, backward selection';
proc logistic data=cow_3 plots=roc;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = backward slstay = 0.05;
run;

title '3�� �� �ٺ��� ������ƽ, stepwise selection';
proc logistic data=cow_3 plots=roc;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = stepwise slentry = 0.05 slstay = 0.05;
run;

title '3�� �� �ٺ��� ������ƽ';
proc logistic data=cow_3 plots=roc;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i  s_t_m  a_s_m_w a_s_m_i  a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit;
run;

/*5�� Multi. Logistic. Regression */

title '5�� �� �ٺ��� ������ƽ';
proc logistic data=cow_5 plots=all;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_t_m a_s_t_m farm_level  / link = glogit;
run;

