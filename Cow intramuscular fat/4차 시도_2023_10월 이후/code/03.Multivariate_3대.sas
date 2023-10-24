/* 한국 변수명 */
options validvarname = any;

/*  SAS Dataset 불러오기 */
libname COW 'C:\Users\Owner\Desktop\윤태준\소\231024_Analysis\data';

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
ods excel file='C:\Users\Owner\Desktop\윤태준\소\231024_Analysis\result\full농가제거mul.xlsx';
/* Logistic regression AUC 0.79*/
proc logistic data=TRAINSET plots=ROC;
class 출산여부_6개월내 (ref='Y') 농가구분 (ref='농가소');
model TARGET1 (event='1') =  개월령 생시체중 근내평균 도체범위근내평균 체고 체장
												형매도축수 형매도체평균 형매근내평균 형매근내평균가산
												출산여부_6개월내 농가근내평균 농가근내평균가산 근내EPD
												농가구분;
run;

ods excel close;
ods listing

/* */

ods listing close;
ods excel file='C:\Users\Owner\Desktop\윤태준\소\231024_Analysis\result\full_mul_.xlsx';
/* Logistic regression for 9 -AUC 0.743*/
proc logistic data=TRAINSET plots=ROC;
class 출산여부_6개월내 (ref='Y') 농가구분 (ref='농가소');
model TARGET1 (event='1') =  개월령 생시체중 근내평균 도체범위근내평균 체고 체장
												형매도축수 형매도체평균 형매근내평균 형매근내평균가산
												출산여부_6개월내 농가근내평균 농가근내평균가산 근내EPD
												농가구분/ firth;
run;

