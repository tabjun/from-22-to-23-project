/* Data Import */
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_3\�ٳ�����3_0.xlsx' out=RAW1 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_3\�ٳ�����3_1.xlsx' out=RAW2 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_3\�ٳ�����3_2.xlsx' out=RAW3 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_3\�ٳ�����3_3.xlsx' out=RAW4 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_3\�ٳ�����3_4.xlsx' out=RAW5 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_3\�ٳ�����3_5.xlsx' out=RAW6 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_3\�ٳ�����3_6.xlsx' out=RAW7 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_3\�ٳ�����3_7.xlsx' out=RAW8 replace; sheet='sheet1'; run;

/* RAW data ���� */
data RAW;
set RAW1 RAW2 RAW3 RAW4 RAW5 RAW6 RAW7 RAW8;
run;

/* ���� Check �׸� ���� */
data RAW;
set RAW;
if ID = '��ǥ��ȣ' then delete;;
run;

/* ������ ������ */
data RAW;
set RAW;
if FARM_CLASS = '1++A' | FARM_CLASS = '1++B' | FARM_CLASS = '1++C' then FARM_LEVEL = 1; else
if FARM_CLASS = '1+A' | FARM_CLASS = '1+B' | FARM_CLASS = '1+C' then FARM_LEVEL = 2; else FARM_LEVEL = 3;

if TARGET = . then TARGET1 = . ; else
if TARGET >= 7 then TARGET1 = 1 ; else TARGET1 = 0;

if TARGET = . then TARGET2 = . ; else
if TARGET >= 9 then TARGET2 = 1 ; else TARGET2 = 0;

if TARGET = . then delete;

if FARM_LEVEL = 1;
run;

proc freq data=RAW;
table TARGET TARGET1 TARGET2;
run;

/* Dataset Split */
proc surveyselect data = RAW method = SRS rep = 1 sampsize = 845 seed = 12345 out = TEST;
  id ID;
run;

proc sort data=RAW; by ID; run;
proc sort data=TEST; by ID; run;

data TESTSET;
merge RAW TEST;
by ID;
if Replicate = 1;
run;

data VALIDSET;
merge RAW TEST;
by ID;
if Replicate ^= 1;
run;

/* SAS Dataset ���� */
libname COW 'C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\230301_Analysis\������';

data COW.RAW; set RAW; run;
data COW.TESTSET; set TESTSET; run;
data COW.VALIDSET; set VALIDSET; run;
