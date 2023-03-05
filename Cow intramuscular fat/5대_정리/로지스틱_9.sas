/* 근내지방 9 -> target 1로 logistic */
libname cow '/home/u62271851/소_최종';

/*** 5대 데이터 불러오기 ***/
/* train */
filename train_9'/home/u62271851/소_최종/5대train_9_최종.csv';
proc import  out = cow.train_5_9
datafile=train_9
dbms=csv
replace;
getnames=yes;
run;

/* validation */
filename test_9 '/home/u62271851/소_최종/5대test_9_최종.csv';
proc import  out = cow.validation_5_9
datafile=test_9
dbms=csv
replace;
getnames=yes;
run;


/* train 기술통계 */
proc means data=cow.train_5_9 N nmiss min max mean std; 
var s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c 
b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c c_s_m_w 
c_s_m_i c_s_f_m c_s_t_m c_s_c d_s_m_w d_s_m_i d_s_f_m d_s_t_m d_s_c ;
run;

proc freq data=cow.train_5_9;
table gender farm_level;
run;

proc means data=cow.train_5_9 n nmiss;
var target;
run;

proc means data=cow.train_5_9 n nmiss;
var target;
run;

/* 단변량 로지스틱 */
ods listing close;
ods pdf file='/home/u62271851/소_최종/5대 단변량 logistic 결과.pdf';

title '성별';
proc logistic data=cow.train_5_9;
class gender(ref='N_M')/param=ref; /* N_M vs F */
model target(event='1') = gender;
run; 

title '도체중';
proc logistic data=cow.train_5_9;
model target(event='1') = s_w ;
run;

title '도축개월';
proc logistic data=cow.train_5_9;
model target(event='1') = sl_m ;
run;

title '형매 도체중 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = s_m_w;
run;

title '형매 등심단면적 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = s_m_i;
run;

title '형매 등지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = s_f_m ;
run;

title '형매 근내지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = s_t_m ;
run;

title '형매 마릿수';
proc logistic data=cow.train_5_9;
model target(event='1') = s_c;
run;

title '어미형매 도체중 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = a_s_m_w;
run;

title '어미형매 등심단면적 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = a_s_m_i ;
run;

title '어미형매 등지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = a_s_f_m;
run;

title '어미형매 근내지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = a_s_t_m;
run;

title '어미형매 마릿수';
proc logistic data=cow.train_5_9;
model target(event='1') = a_s_c;
run;

title '외할미형매 도체중 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = b_s_m_w ;
run;

title '외할미형매 등심단면적 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = b_s_m_i;
run;

title '외할미형매 등지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = b_s_f_m;
run;

title '외할미형매 근내지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = b_s_t_m;
run;

title '외할미형매 마릿수';
proc logistic data=cow.train_5_9;
model target(event='1') = b_s_c ;
run;

title '외증조할미형매 도체중 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = c_s_m_w ;
run;

title '외증조할미형매 등심단면적 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = c_s_m_i;
run;

title '외증조할미형매 등지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = c_s_f_m;
run;

title '외증조할미형매 근내지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = c_s_t_m ;
run;

title '외증조할미형매 마릿수';
proc logistic data=cow.train_5_9;
model target(event='1') = c_s_c ;
run;

title '외고조할미형매 도체중 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = d_s_m_w ;
run;

title '외고조할미형매 등심단면적 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = d_s_m_i ;
run;

title '외고조할미형매 등지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = d_s_f_m ;
run;

title '외고조할미형매 근내지방 평균';
proc logistic data=cow.train_5_9;
model target(event='1') = d_s_t_m ;
run;

title '외고조할미형매 마릿수';
proc logistic data=cow.train_5_9;
model target(event='1') = d_s_c ;
run;

title '농장등급';
proc logistic data=cow.train_5_9;
class farm_level(ref='A')/param=ref;
model target(event='1') = farm_level;
run;


ods pdf close;
ods listing;



/******** Uni&Multi ********/
/*** train set ***/
title1 'Uni&Multi';
proc logistic data=cow.train_5_9 plots(maxpoints=none)=roc;
class gender(ref='N_M') farm_level(ref='A') / param=ref;
model target(event='1') = gender s_w sl_m s_m_i s_t_m a_s_t_m c_s_m_i c_s_t_m d_s_t_m farm_level;
run;


/* 최종 모형 gender sl_m, s_m_i c_s_m_i c_s_t_m d_s_t_m빼고 실행 */
title1 'Uni&Multi';
proc logistic data=cow.train_5_9 plots(maxpoints=none)=roc;
class gender(ref='N_M') farm_level(ref='A') / param=ref;
model target(event='1') = s_w s_t_m farm_level/rocci outroc=troc;
score data=cow.validation_5_9 out=valpred outroc=vroc;
store multiple;
roc; roccontrast; 
run;

/* train 기술 통계 */
proc means data=cow.train_5_9 N nmiss min max mean std; 
var s_w s_t_m;
run;

proc freq data=cow.train_5_9;
table farm_level target;
run;

/* validation 기술 통계 */
proc means data=cow.validation_5_9 N nmiss min max mean std; 
var s_w s_t_m;
run;

proc freq data=cow.validation_5_9;
table farm_level target;
run;


/** cut-off 0.5일 때 **/ 
data valpred1;
set valpred;
if P_1=. then excepted=.;
else if P_1 >= 0.5 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 multiple 결과-cutoff 0.5';
proc freq data=valpred1;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred1(obs=5);
run;


/** cut-off 0.6일 때 **/ 
data valpred2;
set valpred;
if P_1=. then excepted=.;
else if P_1 >= 0.6 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 multiple 결과-cutoff 0.6';
proc freq data=valpred2;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred2(obs=5);
run;


/** cut-off 0.7일 때 **/ 
data valpred3;
set valpred;
if P_1=. then excepted=.;
else if P_1 >= 0.7 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 multiple 결과-cutoff 0.7';
proc freq data=valpred3;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred3(obs=5);
run;


/* cut-off 0.8일 때 */ 
data valpred4;
set valpred;
if P_1=. then excepted=.;
else if P_1 >= 0.8 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 multiple 결과-cutoff 0.8';
proc freq data=valpred4;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred4(obs=5);
run;


/** cut-off 0.9일 때 **/ 
data valpred5;
set valpred;
if P_1=. then excepted=.;
else if P_1 >= 0.9 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 multiple 결과-cutoff 0.9';
proc freq data=valpred5;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred5(obs=5);
run;





/******** 전진선택법 다변량 로지스틱 ********/
/*** train set ***/
title '다변량 로지스틱_전진선택법';
proc logistic data=cow.train_5_9 plots=roc;
class gender(ref='N_M') farm_level(ref='A') / param=ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w 
a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c c_s_m_w 
c_s_m_i c_s_f_m c_s_t_m c_s_c d_s_m_w d_s_m_i d_s_f_m d_s_t_m d_s_c farm_level
/ selection=forward slentry=0.05 details rocci outroc=troc_for;
score data=cow.validation_5_9 out=valpred_for outroc=vroc_for;
roc; roccontrast; 
run;
* farm_level s_w s_t_m a_s_t_m a_s_m_i s_f_m;


/** cut-off 0.5일 때 **/ 
data valpred_for1;
set valpred_for;
if P_1=. then excepted=.;
else if P_1 >= 0.5 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.5';
proc freq data=valpred_for1;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_for1(obs=5);
run;


/** cut-off 0.6일 때 **/ 
data valpred_for2;
set valpred_for;
if P_1=. then excepted=.;
else if P_1 >= 0.6 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.6';
proc freq data=valpred_for2;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_for2(obs=5);
run;


/** cut-off 0.7일 때 **/ 
data valpred_for3;
set valpred_for;
if P_1=. then excepted=.;
else if P_1 >= 0.7 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.7';
proc freq data=valpred_for3;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_for3(obs=5);
run;


/* cut-off 0.8일 때 */ 
data valpred_for4;
set valpred_for;
if P_1=. then excepted=.;
else if P_1 >= 0.8 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.8';
proc freq data=valpred_for4;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_for4(obs=5);
run;


/** cut-off 0.9일 때 **/ 
data valpred_for5;
set valpred_for;
if P_1=. then excepted=.;
else if P_1 >= 0.9 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.9';
proc freq data=valpred_for5;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_for5(obs=5);
run;





/******** 후진소거법 다변량 로지스틱 ********/
title '다변량 로지스틱_전진선택법';
proc logistic data=cow.train_5_9 plots=roc;
class gender(ref='N_M') farm_level(ref='A') / param=ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w 
a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c c_s_m_w 
c_s_m_i c_s_f_m c_s_t_m c_s_c d_s_m_w d_s_m_i d_s_f_m d_s_t_m d_s_c farm_level
/ selection=backward slentry=0.05 details rocci outroc=troc_for;
score data=cow.validation_5_9 out=valpred_back outroc=vroc_back;
roc; roccontrast; 
run;
* farm_level s_w s_t_m a_s_t_m a_s_m_i s_f_m;

/** cut-off 0.5일 때 **/ 
data valpred_back1;
set valpred_back;
if P_1=. then excepted=.;
else if P_1 >= 0.5 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.5';
proc freq data=valpred_back1;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_back1(obs=5);
run;


/** cut-off 0.6일 때 **/ 
data valpred_back2;
set valpred_back;
if P_1=. then excepted=.;
else if P_1 >= 0.6 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.6';
proc freq data=valpred_back2;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_back2(obs=5);
run;


/** cut-off 0.7일 때 **/ 
data valpred_back3;
set valpred_back;
if P_1=. then excepted=.;
else if P_1 >= 0.7 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.7';
proc freq data=valpred_back3;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_back3(obs=5);
run;


/* cut-off 0.8일 때 */ 
data valpred_back4;
set valpred_back;
if P_1=. then excepted=.;
else if P_1 >= 0.8 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.8';
proc freq data=valpred_back4;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_back4(obs=5);
run;


/** cut-off 0.9일 때 **/ 
data valpred_back5;
set valpred_back;
if P_1=. then excepted=.;
else if P_1 >= 0.9 then excepted=1;
else excepted=0;
run;

/* 혼동행렬 그리기 */ 
title1 '5대 forward 결과-cutoff 0.9';
proc freq data=valpred_back5;
tables target*excepted/senspec nocol norow nopercent;
run;

proc print data=valpred_back5(obs=5);
run;





/******** 단계별 선택법 다변량 로지스틱 ********/
/*** train set ***/
title '다변량 로지스틱_전진선택법';
proc logistic data=cow.train_5_9 plots=roc;
class gender(ref='N_M') farm_level(ref='A') / param=ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w 
a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c c_s_m_w 
c_s_m_i c_s_f_m c_s_t_m c_s_c d_s_m_w d_s_m_i d_s_f_m d_s_t_m d_s_c farm_level
/ selection=stepwise slentry=0.05 details rocci outroc=troc_for;
score data=cow.validation_5_9 out=valpred_step outroc=vroc_step;
roc; roccontrast; 
run;
* farm_level s_w s_t_m a_s_t_m a_s_m_i s_f_m;
