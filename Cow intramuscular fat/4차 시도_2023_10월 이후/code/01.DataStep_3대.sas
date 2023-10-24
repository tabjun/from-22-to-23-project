options validvarname = any;

/* Data Import */
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\231024_Analysis\data\sas사용3.csv' out=RAW replace;  run;

proc print data = raw (obs=10);
run;

proc freq data=RAW ;
table TARGET1;
run;

/* Dataset Split */
proc surveyselect data = RAW method = SRS rep = 1 sampsize = 53414 seed = 2023 out = TRAIN;
 id ID;
run;

proc sort data=RAW; by ID; run;
proc sort data=TRAIN; by ID; run;

data TRAINSET;
merge RAW TRAIN;
by ID;
if Replicate = 1;
run;

data VALIDSET;
merge RAW TRAIN;
by ID;
if Replicate ^= 1;
run;

/* SAS Dataset 저장 */
libname COW 'C:\Users\Owner\Desktop\윤태준\소\231024_Analysis\data';

data COW.RAW; set RAW; run;
data COW.TRAINSET; set TRAINSET; run;
data COW.VALIDSET; set VALIDSET; run;
