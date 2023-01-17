libname a 'C:\Users\215-04\Desktop\��_������';
proc import datafile='C:\Users\215-04\Desktop\��_������\cow_total.csv'
dbms=csv
out = a.cow
replace;
getnames=yes;
run;
proc print data= a.cow (obs=10);
run;
/* gender frequency */
proc freq data= a.cow;
tables gender;
run;
DATA cow_gen;
SET a.cow;
IF gender =  '��Ÿ'  THEN DELETE; else
if gender = '����' then delete; else
if gender  = '�̰�' then delete;
RUN;
proc freq data= cow_gen;
tables gender;
run;

/* ��� �������� */
ods listing close;
ods pdf file='C:\Users\215-04\Desktop\��_������\cow.pdf';
/* multinominal logistic */
proc logistic data=a.cow;
class gender (ref='��')/param = ref;
model target =gender sl_m s_w s_m_w s_m_i s_t_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c c_s_m_w
c_s_m_i c_s_f_m c_s_t_m c_s_c d_s_m_w d_s_m_i d_s_f_m d_s_t_m d_s_c / link = glogit;
run;
ods pdf close;
ods listing;

/* ���� ��Ÿ, ������ƾ, �̰� ���� ��, ��, ��, �ż��� ���ܵ� ��*/
/* ��� �������� */
ods listing close;
ods pdf file='C:\Users\215-04\Desktop\��_������\cow_gen.pdf';
/* multinominal logistic */
title1 'gender only ��, ��, �ż�';
proc logistic data=cow_gen;
class gender (ref='��')/param = ref;
model target =gender sl_m s_w s_m_w s_m_i s_t_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c c_s_m_w
c_s_m_i c_s_f_m c_s_t_m c_s_c d_s_m_w d_s_m_i d_s_f_m d_s_t_m d_s_c / link = glogit;
run;
ods pdf close;
ods listing;


