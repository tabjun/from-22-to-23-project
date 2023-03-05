libname cow '/home/u62271851/소';

/*** 3대 데이터 불러오기 ***/
/* train */
filename x1 '/home/u62271851/소/3_train_eng.csv' encoding='utf-8';
proc import  out = cow.train_3
datafile=x1
dbms=csv
replace;
getnames=yes;
run;


/* test */
filename y1 '/home/u62271851/소/3_test_eng.csv' encoding='utf-8';
proc import  out = cow.test_3
datafile=y1
dbms=csv
replace;
getnames=yes;
run;



/*** 5대 데이터 불러오기 ***/
/* train */
filename x2 '/home/u62271851/소/5_train_eng.csv' encoding='utf-8';
proc import  out = cow.train_5
datafile=x2
dbms=csv
replace;
getnames=yes;
run;

/* test */
filename y2 '/home/u62271851/소/5_test_eng.csv' encoding='utf-8';
proc import  out = cow.test_5
datafile=y2
dbms=csv
replace;
getnames=yes;
run;



/*** 3대 로지스틱 회귀분석 ***/
proc logistic data=cow.train_3;
class gender (ref='수') target(ref='1') /param = ref;
model target(event ='1') = gender sl_m s_w s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i 
a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c / link = glogit;
store train3_logistic;
run;

/* test set에 적용 */
proc plm restore=train3_logistic;
score data=cow.test_3 out=cow.test3_scored/ilink;
run;

/* cut-off */
data cow.test3_cut;
set cow.test3_scored;
if predicted > 0.5 then predict= 1;
else predict = 0;
run;

/* sensitivty specificity */
ods listing close;
ods pdf file='/home/u62271851/소/logistic 3대 결과.pdf';
title1 '3대 결과';
proc freq data=cow.test3_cut order=data;
tables target*predict / senspec nocol norow nopercent;
run;
ods pdf close;
ods listing;



/*** 5대 로지스틱 회귀분석 ***/
proc logistic data=cow.train_5;
class gender (ref='수') target(ref='1') /param = ref;
model target = gender sl_m s_w s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i 
a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c / link = glogit;
store train5_logistic;
run;

/* test set에 적용 */
proc plm restore=train5_logistic;
score data=cow.test_5 out=cow.test5_scored/ilink;
run;

/* cut-off */
data cow.test5_cut;
set cow.test5_scored;
if predicted > 0.5 then predict= 1;
else predict = 0;
run;

/* sensitivty specificity */
ods listing close;
ods pdf file='/home/u62271851/소/logistic 5대 결과.pdf';
title1 '5대 결과';
proc freq data=cow.test5_cut order=data;
tables target*predict / senspec nocol norow nopercent;
run;
ods pdf close;
ods listing;












