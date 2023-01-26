libname a 'C:\Users\Owner\Desktop\������\��\������_�����';
/* 3�� �ҷ����� */
proc import datafile='C:\Users\Owner\Desktop\������\��\������_�����\3��� ����\3_train_eng.csv'
dbms=csv
out = a.train_3
replace;
getnames=yes;
run;
proc print data= a.train_3 (obs=10);
run;
proc freq data=a.train_3;
tables gender;
run;
ods listing close;
ods excel file='C:\Users\Owner\Desktop\������\��\������_�����\3��� ����\3�� ������ƽ ���.xlsx';
proc logistic data=a.train_3 plots(maxpoints=none)=all;
class gender (ref='��') target(ref='1') /param = ref;
model target = gender sl_m s_w s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c / link = glogit;
run;
ods excel close;
ods listing;

/* 5�� �ҷ����� */
proc import datafile='C:\Users\Owner\Desktop\������\��\������_�����\5��� ����\5_train.csv'
dbms=csv
out = a.cow_5
replace;
getnames=yes;
run;
proc print data= a.cow_5 (obs=10);
run;
title1 '5�� �� gender only ��, ��, �ż�';
proc freq data= a.cow_5;
tables gender;
run;
proc freq data = a.cow_5;
tables target;
run;

ods listing close;
ods excel file='C:\Users\Owner\Desktop\������\��\������_�����\2022-11-25�ٳ����� 5���\5�� ������ƽ ���.xlsx';
/* multinominal logistic */
title1 'gender only ��, ��, �ż�';
proc logistic data=a.cow_5 plots(maxpoints=none)=all;
class gender (ref='��') target(ref='1') /param = ref;
model target = gender sl_m s_w s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c c_s_m_w
c_s_m_i c_s_f_m c_s_t_m c_s_c d_s_m_w d_s_m_i d_s_f_m d_s_t_m d_s_c / link = glogit;
run;
ods excel close;
ods listing;
