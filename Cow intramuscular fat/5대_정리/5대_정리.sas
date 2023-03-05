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
