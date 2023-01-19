libname a 'C:\Users\Owner\Desktop\������\��\������_�����\5��� ����';
/* 5�� �ҷ����� */
proc import datafile='C:\Users\Owner\Desktop\������\��\������_�����\5��� ����\cow_full.csv'
dbms=csv
out = a.cow_5
replace;
getnames=yes;
run;
proc print data= a.cow (obs=10);
run;
/* ������ ���̺� */
data a;
set a.cow_5;
label gender='����'  sl_m='���ళ��'  s_w='��ü��'  s_m_w='���� ��ü�� ���'  s_m_i='���� ��ɴܸ��� ���'  s_f_m='���� ������ ���'  s_t_m = '���� �ٳ����� ���'  s_c = '���� ������'  a_s_m_w='������� ��ü�� ���'
a_s_m_i='������� ��ɴܸ��� ���'  a_s_f_m='������� ������ ���'  a_s_t_m='������� �ٳ����� ���'  a_s_c='������� ������'  b_s_m_w='���ҹ����� ��ü�� ���'  b_s_m_i='���ҹ����� ��ɴܸ��� ���'  b_s_f_m='���ҹ����� ������ ���'
b_s_t_m='���ҹ����� �ٳ����� ���'  b_s_c='���ҹ����� ������'  c_s_m_w = '�������ҹ����� ��ü�� ���' c_s_m_i = '�������ҹ����� ��ɴܸ��� ���'  c_s_f_m = '�������ҹ����� ������ ���'  c_s_t_m = '�������ҹ����� �ٳ����� ���'
c_s_c = '�������ҹ����� ������'  d_s_m_w='�ܰ����ҹ����� ��ü�� ���'  d_s_m_i='�ܰ����ҹ����� ��ɴܸ��� ���'  d_s_f_m='�ܰ����ҹ����� ������ ���'  d_s_t_m='�ܰ����ҹ����� �ٳ����� ���'  d_s_c = '�ܰ����ҹ����� ������';
run;

/* gender frequency */
title1 '5�� �� gender only ��, ��, �ż�';
proc freq data= a.cow_5;
tables gender;
run;
proc freq data = a.cow_5;
tables target;
run;
/* ��� */
proc freq data = a.cow_5;
tables class;
run;
/* ���� */
proc freq data = a.cow_5;
tables b_c;
run;
/* ����� */
proc freq data = a.cow_5;
tables f_c;
run;
/* ������ */
proc freq data = a.cow_5;
tables m_l;
run;
/* ������ */
proc freq data = a.cow_5;
tables organ;
run;

/* ���� �ż� �� ���� ����� 
DATA cow_gen;
SET a.cow;
IF gender =  '��Ÿ'  THEN DELETE; else
if gender = '����' then delete; else
if gender  = '�̰�' then delete;
RUN;
proc freq data= cow_gen;
tables gender;
run;
*/

/* ���� ��Ÿ, ������ƾ, �̰� ���� ��, ��, ��, �ż��� ���ܵ� ��*/
/* ��� �������� */
ods listing close;
ods pdf file='C:\Users\Owner\Desktop\������\��\������_�����\2022-11-25�ٳ����� 5���\5�� ������ƽ ���.pdf';
/* multinominal logistic */
title1 'gender only ��, ��, �ż�';
proc logistic data=a.cow_5 plots=all;
class gender (ref='��') target(ref='1') /param = ref;
model target = gender sl_m s_w s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c c_s_m_w
c_s_m_i c_s_f_m c_s_t_m c_s_c d_s_m_w d_s_m_i d_s_f_m d_s_t_m d_s_c / link = glogit;
run;
ods pdf close;
ods listing;

proc means data=a.cow_5 n nmiss mean std median q1 q3 min max;
run;

/* 3�� �� */
proc import datafile='C:\Users\Owner\Desktop\������\��\������_�����\2022-11-25�ٳ����� 3���\cow_3.csv'
dbms=csv
out = a.cow_3
replace;
getnames=yes;
run;
proc print data= a.cow_3 (obs=10);
run;
proc freq data= a.cow_3;
tables gender;
run;
DATA cow_gen_3;
SET a.cow_3;
IF gender =  '��Ÿ'  THEN DELETE; else
if gender = '����' then delete; else
if gender  = '�̰�' then delete;
RUN;
title1 '3�� �� ���� �� ����';
proc freq data= cow_gen_3;
tables gender;
run;
title1 '3�� �� �ٳ����浵  �� ����';
proc freq data= a.cow_3;
tables target;
run;
proc freq data= a.cow_3;
tables gender;
run;
/* ��� */
proc freq data = a.cow_3;
tables class;
run;
/* ���� */
proc freq data = a.cow_3;
tables b_c;
run;
/* ����� */
proc freq data = a.cow_3;
tables f_c;
run;
/* ������ */
proc freq data = a.cow_3;
tables m_l;
run;
/* ������ */
proc freq data = a.cow_3;
tables organ;
run;

title1 '3�� �� gender only ��, ��, �ż�';
proc logistic data=cow_gen_3 plots=all;
class gender (ref='��') target(ref='1') /param = ref;
model target = gender sl_m s_w s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c / link = glogit;
run;

proc means data=a.cow_3 n nmiss mean std median q1 q3 min max;
run;
