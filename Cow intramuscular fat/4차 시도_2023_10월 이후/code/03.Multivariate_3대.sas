/* �ѱ� ������ */
options validvarname = any;

/*  SAS Dataset �ҷ����� */
libname COW 'C:\Users\Owner\Desktop\������\��\231024_Analysis\data';

data RAW; set COW.RAW; run;
data TRAINSET; set COW.TRAINSET; run;
data VALIDSET; set COW.VALIDSET; run;

data TRAINSET; 
set TRAINSET; 
run;

proc freq data= trainset;
tables target1;
run;

proc means data= RAW n nmiss min max mean std;
run;

/* */
ods listing close;
ods excel file='C:\Users\Owner\Desktop\������\��\231024_Analysis\result\full������mul.xlsx';
/* Logistic regression AUC 0.79*/
proc logistic data=TRAINSET plots=ROC;
class ��꿩��_6������ (ref='Y') �󰡱��� (ref='�󰡼�');
model TARGET1 (event='1') =  ������ ����ü�� �ٳ���� ��ü�����ٳ���� ü�� ü��
												���ŵ���� ���ŵ�ü��� ���űٳ���� ���űٳ���հ���
												��꿩��_6������ �󰡱ٳ���� �󰡱ٳ���հ��� �ٳ�EPD
												�󰡱���;
run;

ods excel close;
ods listing

/* */

ods listing close;
ods excel file='C:\Users\Owner\Desktop\������\��\231024_Analysis\result\full_mul_.xlsx';
/* Logistic regression for 9 -AUC 0.743*/
proc logistic data=TRAINSET plots=ROC;
class ��꿩��_6������ (ref='Y') �󰡱��� (ref='�󰡼�');
model TARGET1 (event='1') =  ������ ����ü�� �ٳ���� ��ü�����ٳ���� ü�� ü��
												���ŵ���� ���ŵ�ü��� ���űٳ���� ���űٳ���հ���
												��꿩��_6������ �󰡱ٳ���� �󰡱ٳ���հ��� �ٳ�EPD
												�󰡱���/ firth;
run;

