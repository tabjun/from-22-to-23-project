/* 3대 소 단변량 */
/* 소 3대 처리 */ 

/* import full */
filename a 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\소_3대 정리\등급추가_full_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_3_full
replace;
getnames=yes;
run;
proc print data= cow_3_full (obs=5);
run;

/* import train */
filename a 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\소_3대 정리\등급추가_train_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_3_train
replace;
getnames=yes;
run;
proc print data= cow_3_train (obs=5);
run;

/* import test */
filename a 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\소_3대 정리\등급추가_test_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_3_val
replace;
getnames=yes;
run;
proc print data= cow_3_val (obs=5);
run;

/* univariate logistic */
ods listing close;
ods pdf file='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\소_3대 정리\대_소_단변량_결과.pdf';

title '성별';
proc logistic data=cow_3_train;
class gender (ref='N_M') /param = ref;
model target(event='1') = gender  / link = glogit;
run;

title '도축개월';
proc logistic data=cow_3_train;
class target(ref='1')/param = ref;
model target(event='1') = sl_m  / link = glogit;
run;

title '도체중';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_w  / link = glogit;
run;

title '형매 도체중 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_m_w  / link = glogit;
run;

title '형매 등심단면적 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_m_i  / link = glogit;
run;

title '형매 등지방 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_f_m  / link = glogit;
run;

title '형매 근내지방 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_t_m  / link = glogit;
run;

title '형매 마릿수';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = s_c  / link = glogit;
run;

title '어미형매 도체중 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = a_s_m_w  / link = glogit;
run;

title '어미형매 등심단면적 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = a_s_m_i  / link = glogit;
run;

title '어미형매 등지방 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = a_s_f_m  / link = glogit;
run;

title '어미형매 근내지방 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') =a_s_t_m  / link = glogit;
run;

title '어미형매 마릿수';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = a_s_c  / link = glogit;
run;

title '외할미형매 도체중 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = b_s_m_w  / link = glogit;
run;

title '외할미형매 등심단면적 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = b_s_m_i  / link = glogit;
run;

title '외할미형매 등지방 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = b_s_f_m  / link = glogit;
run;

title '외할미형매 근내지방 평균';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') =b_s_t_m  / link = glogit;
run;

title '외할미형매 마릿수';
proc logistic data=cow_3_train;
class /param = ref;
model target(event='1') = b_s_c  / link = glogit;
run;

title '농장 등급';
proc logistic data=cow_3_train;
class farm_level(ref='C') /param = ref;
model target(event='1') = farm_level  / link = glogit;
run;

ods pdf close;
ods listing;

/*3대 Multi. Logistic. Regression */

title '3대 소 다변량 로지스틱, forward selection';
proc logistic data=cow_3_train;
class farm_level(ref='C') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = forward slentry = 0.05;
run;

title '3대 소 다변량 로지스틱, backward selection';
proc logistic data=cow_3_train;
class farm_level(ref='C') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = backward slstay = 0.05;
run;

title '3대 소 다변량 로지스틱, stepwise selection';
proc logistic data=cow_3_train plots=roc;
class farm_level(ref='C') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = stepwise slentry = 0.05 slstay = 0.05;
run;

title '3대 소 다변량 로지스틱, 형매 등지방, 마릿수, 어미형매 등지방 평균 제외';
proc logistic data=cow_3_train plots=roc;
class farm_level(ref='C') gender (ref='N_M') /param = ref;
model target(event='1') =gender s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_m_i a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit;
run;

/* 기초통계량 */
proc means data=cow_3_full n nmiss min max mean std maxdec=1;
class gender;
var s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_m_i a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c;
run;

proc means data=cow_3_train n nmiss min max mean std maxdec=1;
class gender;
var s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_m_i a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c;
run;


proc means data=cow_3_test n nmiss min max mean std maxdec=1;
class gender;
var s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_m_i a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c;
run;

/*3대 소 최종 변수 선택 */
/*train set의 confusion matirx auc,sesi, spec뽑기 */
title1 'Uni&Multi, train에서 validation바로 적용, target:1{7,8,9}';
ods graphics on;
proc logistic data=cow_3_train;
class farm_level(ref='A') / param=ref;
model target(event='1') = s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ outroc=troc;
score data=cow_3_test out=valpred outroc=vroc;
roc; roccontrast;
run;

proc print data=valpred (obs=100);
run;


title1 'Uni&Multi';
proc logistic data=cow_3 plots(maxpoints=none)=roc;
class farm_level(ref='C') /param = ref;
model target(event='1') = s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ rocci;
output out= two p = phat;
store uni_multi;
run;

title1 'Uni&Multi';
proc logistic data=cow_3 plots(maxpoints=none)=roc;
class farm_level(ref='A') /param = ref;
model target(event='1') = s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ link = glogit rocci;
output out= two p = phat;
store uni_multi;
run;

/* train 적용한 모델의 proba_구한다음 phat에 담고, 0.5 cut_off로 잡고 yhat에 담아주기*/ 
data three;
set two;
if phat>0.5 then yhat=1;
else yhat=0;
run;

/* 혼동행렬 그리기 */ 
proc freq data=three;
tables target*yhat/ senspec nocol norow nopercent;
run;

title1 'Variable Selection';
proc logistic data=cow_3 plots(maxpoints=none)=roc ;
class farm_level(ref='A') /param = ref;
model target(event='1') = sl_m s_w s_m_w s_m_i s_f_m s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ link = glogit rocci;
output out= two p = phat;
run;

/* train 적용한 모델의 proba_구한다음 phat에 담고, 0.5 cut_off로 잡고 yhat에 담아주기*/ 
data three;
set two;
if phat>0.5 then yhat=1;
else yhat=0;
run;

/* 혼동행렬 그리기 */ 
proc freq data=three;
tables target*yhat/ senspec nocol norow nopercent measures;
run;

title1 'Uni&Multi';
proc logistic data=cow_3;
class farm_level(ref='C') /param = ref;
model target(event='1') = s_w sl_m s_m_w s_m_i s_t_m a_s_m_w a_s_t_m b_s_m_w 
b_s_t_m farm_level/ link = glogit rocci;
output out= two p = phat;
store uni_multi;
run;

/* train 적용한 모델의 proba_구한다음 phat에 담고, 0.5 cut_off로 잡고 yhat에 담아주기*/ 
data three;
set two;
if phat>0.5 then yhat=1;
else yhat=0;
run;

/* 혼동행렬 그리기 */ 
proc freq data=three;
tables target*yhat/ senspec nocol norow nopercent measures;
run;

/* validation 적용 */
proc plm restore=uni_multi;
score data=cow_3_test out= val_3 / ilink;
run;

proc print data=val_3 (obs=5);
run;

/* 0.5 cut-off 지정 */
data val_3_c;
set val_3;
if predicted > 0.5 then predict= 1;
else predict = 0;
run;

title1 '3대 multiple 결과';
proc freq data=val_3_c order=data;
tables target*predict /senspec nocol norow nopercent;
run;


/* selection validation 적용 */
proc plm restore=selection;
score data=cow_3_test out= val_3 / ilink;
run;

proc print data=val_3 (obs=10);
run;

/* 0.5 cut-off 지정 */
data val_3_c;
set val_3;
if predicted > 0.5 then predict= 1;
else predict = 0;
run;

title1 '3대 selection 결과';
proc freq data=val_3_c order=data;
tables target*predict /senspec nocol norow nopercent;
run;
