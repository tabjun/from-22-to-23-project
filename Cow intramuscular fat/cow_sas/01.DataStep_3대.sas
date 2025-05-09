/* Data Import */
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\2023-02-07근내지방_3\근내지방3_0.xlsx' out=RAW1 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\2023-02-07근내지방_3\근내지방3_1.xlsx' out=RAW2 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\2023-02-07근내지방_3\근내지방3_2.xlsx' out=RAW3 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\2023-02-07근내지방_3\근내지방3_3.xlsx' out=RAW4 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\2023-02-07근내지방_3\근내지방3_4.xlsx' out=RAW5 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\2023-02-07근내지방_3\근내지방3_5.xlsx' out=RAW6 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\2023-02-07근내지방_3\근내지방3_6.xlsx' out=RAW7 replace; sheet='sheet1'; run;
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\2023-02-07근내지방_3\근내지방3_7.xlsx' out=RAW8 replace; sheet='sheet1'; run;

/* RAW data 통합 */
data RAW;
set RAW1 RAW2 RAW3 RAW4 RAW5 RAW6 RAW7 RAW8;
run;

/* 변수 Check 항목 삭제 */
data RAW;
set RAW;
if ID = '이표번호' then delete;;
run;

/* 농가정보 나누기 */
data RAW;
set RAW;
if FARM_CLASS = '1++A' | FARM_CLASS = '1++B' | FARM_CLASS = '1++C' then FARM_LEVEL = 1; else
if FARM_CLASS = '1+A' | FARM_CLASS = '1+B' | FARM_CLASS = '1+C' then FARM_LEVEL = 2; else FARM_LEVEL = 3;

if TARGET = . then TARGET1 = . ; else
if TARGET >= 7 then TARGET1 = 1 ; else TARGET1 = 0;

if TARGET = . then TARGET2 = . ; else
if TARGET >= 9 then TARGET2 = 1 ; else TARGET2 = 0;

if TARGET = . then delete;
run;

proc freq data=RAW;
table TARGET TARGET1 TARGET2;
run;

/* Dataset Split */
proc surveyselect data = RAW method = SRS rep = 1 sampsize = 77120 seed = 12345 out = TEST;
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

/* SAS Dataset 저장 */
libname COW 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis';

data COW.RAW; set RAW; run;
data COW.TESTSET; set TESTSET; run;
data COW.VALIDSET; set VALIDSET; run;

/* baseline chracteristics */
proc means data= RAW N NMISS MIN MAX MEAN STD;
var SL_M S_W
      S_M_W S_F_M S_T_M S_C
      A_S_M_W A_S_T_M  
	  B_S_M_W B_S_M_I B_S_T_M; 
run;

proc freq data=RAW;
tables GENDER FARM_LEVEL TARGET1 TARGET2;
RUN;

/* baseline chracteristics */
proc means data= TESTSET N NMISS MIN MAX MEAN STD;
var SL_M S_W
      S_M_W S_F_M S_T_M S_C
      A_S_M_W A_S_T_M  
	  B_S_M_W B_S_M_I B_S_T_M; 
run;

proc freq data=TESTSET;
tables GENDER FARM_LEVEL TARGET1 TARGET2;
RUN;

/* baseline chracteristics */
proc means data= VALIDSET N NMISS MIN MAX MEAN STD;
var SL_M S_W
      S_M_W S_F_M S_T_M S_C
      A_S_M_W A_S_T_M  
	  B_S_M_W B_S_M_I B_S_T_M; 
run;

proc freq data=VALIDSET;
tables GENDER FARM_LEVEL TARGET1 TARGET2;
RUN;
