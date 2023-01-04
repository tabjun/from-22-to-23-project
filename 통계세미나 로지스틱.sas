proc import 

datafile = "/home/u58164541/logistic.csv"

out = dataset

dbms = csv;

run;
proc contents data= dataset;
run;

data dataset;
set dataset;
if incomeq = . then inc_g = .; else
if incomeq <= 3 then inc_g = 1; else
if incomeq <= 7 then inc_g = 2; else inc_g = 3;

if econstat =. then econstat = 3;

if age < 20 or age >50 then delete;

if age <= 30 then age_g = 1 ; else /* 21 ~ 30*/
if age <= 40 then age_g = 2 ; /* 31 ~ 40 */ else age_g=3/* 40~ 49*/;

if inc1 <= 100 then inc1_g = 1 ; else 
if inc1 <= 230 then inc1_g = 2; else
if inc1 <= 450 then inc1_g = 3; else inc1_g = 4;

if inc4 <= 3600 then inc4_g = 1; else
if inc4 <= 5200 then inc4_g = 2; else
if inc4 <= 7200 then inc4_g = 3; else inc4_g = 4;
run;
proc freq data=dataset;
table employtype*p_married;
run;

proc freq data=dataset;
table inc_g*p_married;
run;

proc freq data=dataset;
table age;
run;

/* univariate */
title1 'age, male';
PROC LOGISTIC DATA = dataset;
where sex=1;
class age_g(ref='1');
MODEL p_married (event = '1') = age_g;
RUN;

title1 'age, female';
PROC LOGISTIC DATA = dataset;
where sex=2;
class age_g(ref='1');
MODEL p_married (event = '1') = age_g;
RUN;

title1 'univariate logistic regression';
title2 'incomeq, category, male';
PROC LOGISTIC DATA = dataset;
where sex=1;
CLASS inc_g (ref = '1');
MODEL p_married (event = '1') = inc_g;
RUN;

title2 'incomeq, category, female';
PROC LOGISTIC DATA = dataset;
where sex=2;
CLASS inc_g (ref = '1');
MODEL p_married (event = '1') = inc_g;
RUN;

title2 'inc_1, category, male';
PROC LOGISTIC DATA = dataset;
where sex=1;
CLASS inc1_g (ref = '1');
MODEL p_married (event = '1') = inc1_g;
RUN;

title2 'incomeq, category, female';
PROC LOGISTIC DATA = dataset;
where sex=2;
CLASS inc1_g (ref = '1');
MODEL p_married (event = '1') = inc1_g;
RUN;

title2 'econstat, category, male';
PROC LOGISTIC DATA = dataset;
where sex=1;
CLASS econstat (ref = '2');
MODEL p_married (event ='1') = econstat;
RUN;

title2 'econstat, category, female';
PROC LOGISTIC DATA = dataset;
where sex=2;
CLASS econstat (ref = '2');
MODEL p_married (event ='1') = econstat;
RUN;

title2 'employtype, category, male';
PROC LOGISTIC DATA = dataset;
where sex=1;
CLASS employtype (ref = '1');
MODEL p_married (event = '1')= employtype;
RUN;

title2 'employtype, category, female';
PROC LOGISTIC DATA = dataset;
where sex=2;
CLASS employtype (ref = '3');
MODEL p_married (event = '1')= employtype;
RUN;

title2 'h2602_1, category, male';
PROC LOGISTIC DATA = dataset;
where sex = 1;
CLASS h2602_1 (ref = '0');
MODEL p_married (event = '1') = h2602_1;
RUN;

title2 'h2602_1, category, female';
PROC LOGISTIC DATA = dataset;
where sex = 2;
CLASS h2602_1 (ref = '0');
MODEL p_married (event = '1') = h2602_1;
RUN;

title2 'h2651, male';
PROC LOGISTIC DATA = dataset;
where sex =1;
CLASS h2651 (ref = '1');
MODEL p_married (event = '1') = h2651;
RUN;

title2 'h2651, female';
PROC LOGISTIC DATA = dataset;
where sex =2;
CLASS h2651 (ref = '1');
MODEL p_married (event = '1') = h2651;
RUN;

title2 'oecd, male';
PROC LOGISTIC DATA = dataset;
where sex=1;
MODEL p_married(event = '1') = oecd;
RUN;

title2 'oecd, female';
PROC LOGISTIC DATA = dataset;
where sex=2;
MODEL p_married(event = '1') = oecd;
RUN;

title2 'resid_type, category, male';
PROC LOGISTIC DATA = dataset;
class resid_type(ref= '3');
where sex=1;
MODEL p_married(event = '1') = resid_type;
RUN;

title2 'resid_type, category, female';
PROC LOGISTIC DATA = dataset;
class resid_type(ref= '3');
where sex=2;
MODEL p_married(event = '1') = resid_type;
RUN;

/* multi */
/* 
PROC LOGISTIC DATA = dataset;
CLASS inc_g (ref = '1') econstat (ref = '2') h2602_1(ref= '1') h2651 (ref = '1') resid_type(ref='3') sex(ref='1');
MODEL p_married(event = '1') =age sex inc_g econstat h2602_1 h2651 oecd resid_type;
RUN; */

title1 'multivariate logistic';
title2 'male';
PROC LOGISTIC DATA = dataset;
where sex=1;
CLASS inc_g (ref = '1') age_g(ref='3') econstat (ref = '2') h2602_1(ref= '0') h2651 (ref = '1') resid_type(ref='3');
MODEL p_married(event = '1') =age_g inc_g econstat h2602_1 h2651 oecd resid_type;
RUN;

title2 'female';
PROC LOGISTIC DATA = dataset;
where sex=2;
CLASS inc_g (ref = '1') age_g(ref='3') econstat (ref = '2') h2602_1(ref= '0') h2651 (ref = '1') resid_type(ref='3');
MODEL p_married(event = '1') =age_g inc_g econstat h2602_1 h2651 oecd resid_type;
RUN;