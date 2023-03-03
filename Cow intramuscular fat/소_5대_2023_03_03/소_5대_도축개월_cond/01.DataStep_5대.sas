/* Data Import */
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_5\�ٳ�����5_0.xlsx' out=RAW1 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_5\�ٳ�����5_1.xlsx' out=RAW2 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_5\�ٳ�����5_2.xlsx' out=RAW3 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_5\�ٳ�����5_3.xlsx' out=RAW4 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_5\�ٳ�����5_4.xlsx' out=RAW5 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_5\�ٳ�����5_5.xlsx' out=RAW6 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_5\�ٳ�����5_6.xlsx' out=RAW7 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_5\�ٳ�����5_7.xlsx' out=RAW8 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\2023-02-07�ٳ�����_5\�ٳ�����5_8.xlsx' out=RAW9 replace; sheet='sheet1'; run;


/* RAW data ���� */
data RAW;
set RAW1 RAW2 RAW3 RAW4 RAW5 RAW6 RAW7 RAW8 RAW9;
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

if 25 <= SL_M <= 40;
run;

proc freq data=RAW;
table TARGET TARGET1 TARGET2;
run;

/* Dataset Split */
proc surveyselect data = RAW method = SRS rep = 1 sampsize = 2581 seed = 12345 out = TEST;
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
libname COW 'C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\230301_Analysis_5��\age����';

data COW.RAW; set RAW; run;
data COW.TESTSET; set TESTSET; run;
data COW.VALIDSET; set VALIDSET; run;
