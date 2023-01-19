libname a 'C:\Users\Owner\Desktop\윤태준\소\윤태준_김다은\5계대 정리';
/* 5대 불러오기 */
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\윤태준_김다은\5계대 정리\cow_full.csv'
dbms=csv
out = a.cow_5
replace;
getnames=yes;
run;
proc print data= a.cow (obs=10);
run;
/* 데이터 레이블 */
data a;
set a.cow_5;
label gender='성별'  sl_m='도축개월'  s_w='도체중'  s_m_w='형매 도체중 평균'  s_m_i='형매 등심단면적 평균'  s_f_m='형매 등지방 평균'  s_t_m = '형매 근내지방 평균'  s_c = '형매 마릿수'  a_s_m_w='어미형태 도체중 평균'
a_s_m_i='어미형매 등심단면적 평균'  a_s_f_m='어미형매 등지방 평균'  a_s_t_m='어미형매 근내지방 평균'  a_s_c='어미형매 마릿수'  b_s_m_w='외할미형매 도체중 평균'  b_s_m_i='외할미형매 등심단면적 평균'  b_s_f_m='외할미형매 등지방 평균'
b_s_t_m='외할미형매 근내지방 평균'  b_s_c='외할미형매 마릿수'  c_s_m_w = '외증조할미형매 도체중 평균' c_s_m_i = '외증조할미형매 등심단면적 평균'  c_s_f_m = '외증조할미형매 등지방 평균'  c_s_t_m = '외증조할미형매 근내지방 평균'
c_s_c = '외증조할미형매 마릿수'  d_s_m_w='외고조할미형매 도체중 평균'  d_s_m_i='외고조할미형매 등심단면적 평균'  d_s_f_m='외고조할미형매 등지방 평균'  d_s_t_m='외고조할미형매 근내지방 평균'  d_s_c = '외고조할미형매 마릿수';
run;

/* gender frequency */
title1 '5대 소 gender only 암, 수, 거세';
proc freq data= a.cow_5;
tables gender;
run;
proc freq data = a.cow_5;
tables target;
run;
/* 등급 */
proc freq data = a.cow_5;
tables class;
run;
/* 육색 */
proc freq data = a.cow_5;
tables b_c;
run;
/* 지방색 */
proc freq data = a.cow_5;
tables f_c;
run;
/* 성숙도 */
proc freq data = a.cow_5;
tables m_l;
run;
/* 조직도 */
proc freq data = a.cow_5;
tables organ;
run;

/* 성별 거세 암 수만 남기기 
DATA cow_gen;
SET a.cow;
IF gender =  '기타'  THEN DELETE; else
if gender = '프리' then delete; else
if gender  = '미경' then delete;
RUN;
proc freq data= cow_gen;
tables gender;
run;
*/

/* 성별 기타, 프리마틴, 미경 제외 후, 암, 수, 거세만 남겨둔 것*/
/* 결과 내보내기 */
ods listing close;
ods pdf file='C:\Users\Owner\Desktop\윤태준\소\윤태준_김다은\2022-11-25근내지방 5계대\5대 로지스틱 결과.pdf';
/* multinominal logistic */
title1 'gender only 암, 수, 거세';
proc logistic data=a.cow_5 plots=all;
class gender (ref='수') target(ref='1') /param = ref;
model target = gender sl_m s_w s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c c_s_m_w
c_s_m_i c_s_f_m c_s_t_m c_s_c d_s_m_w d_s_m_i d_s_f_m d_s_t_m d_s_c / link = glogit;
run;
ods pdf close;
ods listing;

proc means data=a.cow_5 n nmiss mean std median q1 q3 min max;
run;

/* 3대 소 */
proc import datafile='C:\Users\Owner\Desktop\윤태준\소\윤태준_김다은\2022-11-25근내지방 3계대\cow_3.csv'
dbms=csv
out = a.cow_3
replace;
getnames=yes;
run;
proc print data= a.cow_3 (obs=10);
run;
proc freq data= a.cow_3;
tables gender;
run;
DATA cow_gen_3;
SET a.cow_3;
IF gender =  '기타'  THEN DELETE; else
if gender = '프리' then delete; else
if gender  = '미경' then delete;
RUN;
title1 '3대 소 성별 값 분포';
proc freq data= cow_gen_3;
tables gender;
run;
title1 '3대 소 근내지방도  값 분포';
proc freq data= a.cow_3;
tables target;
run;
proc freq data= a.cow_3;
tables gender;
run;
/* 등급 */
proc freq data = a.cow_3;
tables class;
run;
/* 육색 */
proc freq data = a.cow_3;
tables b_c;
run;
/* 지방색 */
proc freq data = a.cow_3;
tables f_c;
run;
/* 성숙도 */
proc freq data = a.cow_3;
tables m_l;
run;
/* 조직도 */
proc freq data = a.cow_3;
tables organ;
run;

title1 '3대 소 gender only 암, 수, 거세';
proc logistic data=cow_gen_3 plots=all;
class gender (ref='수') target(ref='1') /param = ref;
model target = gender sl_m s_w s_m_w s_m_i s_f_m s_t_m s_c a_s_m_w a_s_m_i a_s_f_m a_s_t_m a_s_c b_s_m_w b_s_m_i b_s_f_m b_s_t_m b_s_c / link = glogit;
run;

proc means data=a.cow_3 n nmiss mean std median q1 q3 min max;
run;
