/* Data Import */
libname COW 'C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\230504_Analysis\dataset';

data RAW; set COW.RAW; run;

/* KPN ��� ���� ���ڷ� ���� */
data RAW;
set RAW;
if KPN_MEAN_CLASS = 'A' then KPN_MEAN_LEVEL = 3; else
if KPN_MEAN_CLASS = 'B' then KPN_MEAN_LEVEL = 2; else KPN_MEAN_LEVEL = 1;

if KPN_MEDIAN_CLASS = 'A' then KPN_MEDIAN_LEVEL = 3; else
if KPN_MEDIAN_CLASS = 'B' then KPN_MEDIAN_LEVEL = 2; else KPN_MEDIAN_LEVEL = 1;

if 28 <= SL_M <= 35;
run;

proc freq data=RAW;
table KPN_MEAN_CLASS KPN_MEAN_LEVEL KPN_MEDIAN_CLASS KPN_MEDIAN_LEVEL;
run;

/* �� ǥ�� 75007��, 7:3 SPLIT, TRAIN: 52505�� */ 
/* Dataset Split */
proc surveyselect data = RAW method = SRS rep = 1 sampsize = 52505 seed = 12345 out = VALID;
  id ID;
run;

PROC PRINT DATA=RAW(OBS=10);
RUN;

proc sort data=RAW; by ID; run;
proc sort data=VALID; by ID; run;

data TRAIN;
merge RAW VALID;
by ID;
if Replicate = 1;
run;

data VALID;
merge RAW VALID;
by ID;
if Replicate ^= 1;
run;

/* SAS Dataset ���� */
libname COW 'C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\230504_Analysis\Analysis_0515\���ళ��\Data';

data COW.RAW; set RAW; run;
data COW.TRAINSET; set TRAIN; run;
data COW.VALIDSET; set VALID; run;

PROC EXPORT DATA=RAW
OUTFILE='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\230504_Analysis\Analysis_0515\���ళ��\Data\raw.csv'
DBMS=CSV REPLACE;
RUN;

PROC EXPORT DATA=TRAIN
OUTFILE='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\230504_Analysis\Analysis_0515\���ళ��\Data\train.csv'
DBMS=CSV REPLACE;
RUN;

PROC EXPORT DATA=VALID
OUTFILE='C:\Users\Owner\Desktop\������\��\3�� ��û�ڷ�\230504_Analysis\Analysis_0515\���ళ��\Data\valid.csv'
DBMS=CSV REPLACE;
RUN;
