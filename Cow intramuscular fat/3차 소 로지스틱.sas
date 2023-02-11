/* 소 5대 처리 */ 
filename a 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\소_5대 정리\등급추가_train_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_5
replace;
getnames=yes ;
run;
proc print data= cow_5 (obs=5);
run;

/* univariate logistic */
ods listing close;
ods pdf file='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\소_5대 정리\5대_소_단변량_결과.pdf';

title '성별';
proc logistic data=cow_5 plots=roc;
class gender (ref='N_M') /param = ref;
model target(event='1') = gender  / link = glogit;
run;

title '도축개월';
proc logistic data=cow_5;
class target(ref='1')/param = ref;
model target(event='1') = sl_m  / link = glogit;
run;

title '도체중';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_w  / link = glogit;
run;

title '형매 도체중 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_m_w  / link = glogit;
run;

title '형매 등심단면적 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_m_i  / link = glogit;
run;

title '형매 등지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_f_m  / link = glogit;
run;

title '형매 근내지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_t_m  / link = glogit;
run;

title '형매 마릿수';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = s_c  / link = glogit;
run;

title '어미형매 도체중 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = a_s_m_w  / link = glogit;
run;

title '어미형매 등심단면적 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = a_s_m_i  / link = glogit;
run;

title '어미형매 등지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = a_s_f_m  / link = glogit;
run;

title '어미형매 근내지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') =a_s_t_m  / link = glogit;
run;

title '어미형매 마릿수';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = a_s_c  / link = glogit;
run;

title '외할미형매 도체중 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_m_w  / link = glogit;
run;

title '외할미형매 등심단면적 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_m_i  / link = glogit;
run;

title '외할미형매 등지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_f_m  / link = glogit;
run;

title '외할미형매 근내지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_t_m  / link = glogit;
run;

title '외할미형매 마릿수';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = b_s_c  / link = glogit;
run;

title '외증조형매 도체중 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_m_w  / link = glogit;
run;

title '외증조형매 등심단면적 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_m_i  / link = glogit;
run;

title '외증조형매 등지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_f_m  / link = glogit;
run;

title '외증조형매 근내지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_t_m  / link = glogit;
run;

title '외증조형매 마릿수';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = c_s_c  / link = glogit;
run;

title '외고조형매 도체중 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_m_w  / link = glogit;
run;

title '외고조형매 등심단면적 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_m_i  / link = glogit;
run;

title '외고조형매 등지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_f_m  / link = glogit;
run;

title '외고조형매 근내지방 평균';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_t_m  / link = glogit;
run;

title '외고조형매 마릿수';
proc logistic data=cow_5;
class /param = ref;
model target(event='1') = d_s_c  / link = glogit;
run;

title '농장 등급';
proc logistic data=cow_5;
class farm_level(ref='C') /param = ref;
model target(event='1') = farm_level  / link = glogit;
run;

ods pdf close;
ods listing;
/* 3대 소 단변량 */
/* 소 3대 처리 */ 
filename a 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\소_3대 정리\등급추가_train_cp.csv' encoding='cp949';
proc import datafile= a
dbms=csv
out = cow_3
replace;
getnames=yes;
run;
proc print data= cow_3 (obs=10);
run;

/* univariate logistic */
ods listing close;
ods pdf file='C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\소_3대 정리\대_소_단변량_결과.pdf';

title '성별';
proc logistic data=cow_3;
class gender (ref='N_M') /param = ref;
model target(event='1') = gender  / link = glogit;
run;

title '도축개월';
proc logistic data=cow_3;
class target(ref='1')/param = ref;
model target(event='1') = sl_m  / link = glogit;
run;

title '도체중';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_w  / link = glogit;
run;

title '형매 도체중 평균';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_m_w  / link = glogit;
run;

title '형매 등심단면적 평균';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_m_i  / link = glogit;
run;

title '형매 등지방 평균';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_f_m  / link = glogit;
run;

title '형매 근내지방 평균';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_t_m  / link = glogit;
run;

title '형매 마릿수';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = s_c  / link = glogit;
run;

title '어미형매 도체중 평균';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = a_s_m_w  / link = glogit;
run;

title '어미형매 등심단면적 평균';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = a_s_m_i  / link = glogit;
run;

title '어미형매 등지방 평균';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = a_s_f_m  / link = glogit;
run;

title '어미형매 근내지방 평균';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') =a_s_t_m  / link = glogit;
run;

title '어미형매 마릿수';
proc logistic data=cow_3;
class /param = ref;
model target(event='1') = a_s_c  / link = glogit;
run;

title '농장 등급';
proc logistic data=cow_3;
class farm_level(ref='C') /param = ref;
model target(event='1') = farm_level  / link = glogit;
run;

ods pdf close;
ods listing;

/*3대 Multi. Logistic. Regression */

title '3대 소 다변량 로지스틱, forward selection';
proc logistic data=cow_3 plots=roc;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = forward slentry = 0.05;
run;

title '3대 소 다변량 로지스틱, backward selection';
proc logistic data=cow_3 plots=roc;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = backward slstay = 0.05;
run;

title '3대 소 다변량 로지스틱, stepwise selection';
proc logistic data=cow_3 plots=roc;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit selection = stepwise slentry = 0.05 slstay = 0.05;
run;

title '3대 소 다변량 로지스틱';
proc logistic data=cow_3 plots=roc;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i  s_t_m  a_s_m_w a_s_m_i  a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m 
b_s_t_m b_s_c farm_level/ link = glogit;
run;

/*5대 Multi. Logistic. Regression */

title '5대 소 다변량 로지스틱';
proc logistic data=cow_5 plots=all;
class farm_level(ref='A') gender (ref='N_M') /param = ref;
model target(event='1') = gender s_w sl_m s_m_w s_m_i s_t_m a_s_t_m farm_level  / link = glogit;
run;

