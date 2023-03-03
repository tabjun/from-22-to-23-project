/*  SAS Dataset 불러오기 */
libname COW 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis\농장등급';

data RAW; set COW.RAW; run;
data TESTSET; set COW.TESTSET; run;
data VALIDSET; set COW.VALIDSET; run;

* GENDER S_M_W S_M_I S_F_M S_C A_S_M_W A_S_M_I A_S_F_M A_S_T_M A_S_C B_S_M_W B_S_M_I B_S_F_M B_S_T_M B_S_C 제외 (p-value 0.2 기준) - AUC 0.628 - 최종 모형 확인 ;
proc logistic data=TESTSET plots=ROC outest=betas;
class FARM_LEVEL GENDER (ref='거세');
model TARGET1 (event='1') = FARM_LEVEL SL_M S_W
                                               S_T_M
                                               ;
run;

* GENDER SL_M S_M_W S_M_I S_F_M A_S_M_W A_S_F_M A_S_T_M B_S_M_W B_S_M_I B_S_F_M B_S_T_M 제외 (p-value 0.2 기준) - AUC 0.625 - 최종 모형 확인 ;
proc logistic data=TESTSET plots=ROC outest=betas_9;
class FARM_LEVEL  GENDER (ref='거세');
model TARGET2 (event='1') =  FARM_LEVEL S_W
                                               S_T_M
                                               A_S_M_I  
											   / firth;
run;


/* Macro Part */
*----diagnostic performance(sen, spe, acc, ppv, npv & 95% CI);
%macro diag(gs, var, data) ;
proc sort data = &DATA ; by descending &GS; run;
proc freq data = &data order=data; table &var * &gs / out=table0 ; run ;

* 0,1 coding, 1,2 coding에 따라 바꿔줌;
data table ;
	set table0 ;
	if &var   = 0 and &gs  = 0 then class = 'TN' ;
	else if &var   = 0 and &gs  = 1 then class = 'FN' ;
	else if &var   = 1 and &gs  = 0 then class = 'FP' ;
	else if &var   = 1 and &gs  = 1 then class = 'TP' ;
run ;

proc transpose data = table out = report0 ;
	var count ;
	id class;
run ;

data report ;
	set report0 ;
	if TP = . then TP = 0 ;
	if TN = . then TN = 0 ;
	if FP = . then FP = 0 ;
	if FN = . then FN = 0 ;
N=sum(TP,TN,FP,FN) ;

Sensitivity=(TP/(TP+FN))*100;
Sensitivity_l=(Sensitivity/100-1.96*sqrt( (Sensitivity/100)*(1-(Sensitivity/100))/(TP+FN) ) )*100;
Sensitivity_u=(Sensitivity/100+1.96*sqrt( (Sensitivity/100)*(1-(Sensitivity/100))/(TP+FN) ) )*100;

Specificity=(1-FP/(TN+FP))*100;
Specificity_l=(Specificity/100-1.96*sqrt( (Specificity/100)*(1-(Specificity/100))/(FP+TN) ) )*100;
Specificity_u=(Specificity/100+1.96*sqrt( (Specificity/100)*(1-(Specificity/100))/(FP+TN) ) )*100;

Accuracy=((TP+TN)/(TP+TN+FP+FN))*100;
Accuracy_l=(Accuracy/100-1.96*sqrt( (Accuracy/100)*(1-(Accuracy/100))/N ) )*100;
Accuracy_u=(Accuracy/100+1.96*sqrt( (Accuracy/100)*(1-(Accuracy/100))/N ) )*100;

PPV= (TP/(TP+FP))*100 ;
PPV_l=(PPV/100-1.96*sqrt( (PPV/100)*(1-(PPV/100))/(TP+FP) ) )*100;
PPV_u=(PPV/100+1.96*sqrt( (PPV/100)*(1-(PPV/100))/(TP+FP) ) )*100;

NPV= (TN/(TN+FN))*100 ;
NPV_l=(NPV/100-1.96*sqrt( (NPV/100)*(1-(NPV/100))/(TN+FN) ) )*100;
NPV_u=(NPV/100+1.96*sqrt( (NPV/100)*(1-(NPV/100))/(TN+FN) ) )*100;
run ;

proc print data = report ;
	var TP TN FP FN Sensitivity Sensitivity_l Sensitivity_u Specificity Specificity_l Specificity_u 
			accuracy accuracy_l accuracy_u PPV PPV_l PPV_u NPV NPV_l NPV_u ;
run ;
%mend ;



/* Prediction model Validation set  적용 */
data VALIDSET;
set VALIDSET;

if GENDER = '거세' then SEX = 0; else SEX = 1;
if FARM_LEVEL = 1 then F1 = 1;  
if FARM_LEVEL = 2 then F1 = 0;  
if FARM_LEVEL = 3 then F1 = 0;  

if FARM_LEVEL = 1 then F2 = 0;  
if FARM_LEVEL = 2 then F2 = 1;  
if FARM_LEVEL = 3 then F2 = 0;  
run;

data VALIDSET_1;
set VALIDSET;

/* target 789 */
PR = -3.6946 + (0.0486*SL_M) + (0.00634*S_W) + (0.086*S_T_M);

PREDP = exp(PR)/(1+exp(PR));

if PREDP >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP >= 0.9 then PRED9 =1; else PRED9 = 0;

run;

proc print data=VALIDSET_1 (OBS=10);
run;

data VALIDSET_2;
set VALIDSET;

/* target only 9 */
PR_2 = -2.1175 + (0.0057*S_W) + (0.0951*S_T_M) + (-0.0173*A_S_M_I);

PREDP_2 = exp(PR_2)/(1+exp(PR_2));

if PREDP_2 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_2 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_2 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_2 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_2 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_2(OBS=10);
RUN;

/* Performance Comparisons */
data VALIDSET1;
set VALIDSET_1;
keep ID TARGET1 PREDP PRED5 PRED6 PRED7 PRED8 PRED9;
run;

data VALIDSET2;
set VALIDSET_2;
keep ID TARGET2 PREDP_2 PRED5 PRED6 PRED7 PRED8 PRED9;
run;

/* Using Macro */
/* TARGET 789 */
TITLE1 'TARGET 789';
TITLE2 'CUT-OFF 0.5'; /* sensitiviy 51.87 */
%diag(TARGET1, PRED5, VALIDSET1);
TITLE2 'CUT-OFF 0.6'; /* sensitivity 25.08 */
%diag(TARGET1, PRED6, VALIDSET1);
TITLE2 'CUT-OFF 0.7'; /* sensitivity 7.96 */
%diag(TARGET1, PRED7, VALIDSET1);
TITLE2 'CUT-OFF 0.8'; /* sensitivity 2.44 */
%diag(TARGET1, PRED8, VALIDSET1);
TITLE2 'CUT-OFF 0.9'; /* sensitivity 0.51 */
%diag(TARGET1, PRED9, VALIDSET1);

/* TARGET ONLY 9 */
TITLE1 'TARGET ONLY 9';
TITLE2 'CUT-OFF 0.5'; /* sensitiviy 0.14 */
%diag(TARGET2, PRED5, VALIDSET2);
TITLE2 'CUT-OFF 0.6'; /* sensitiviy 0.04 */
%diag(TARGET2, PRED6, VALIDSET2);
TITLE2 'CUT-OFF 0.7'; /* sensitiviy 0.01 */
%diag(TARGET2, PRED7, VALIDSET2);
TITLE2 'CUT-OFF 0.8'; /* sensitiviy 0 */
%diag(TARGET2, PRED8, VALIDSET2);
TITLE2 'CUT-OFF 0.9'; /* sensitiviy 0 */
%diag(TARGET2, PRED9, VALIDSET2);

/* target_789, proba_ desending */
proc sort data=VALIDSET1 OUT=SORTED_1;
BY DESENDING PREDP;
RUN;
PROC PRINT DATA= SORTED_1 (OBS=10);
RUN;

PROC EXPORT DATA=SORTED_1
OUTFILE = 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis\농장등급\sorted_789_result.csv'
dbms=csv replace;
run;

proc sort data=VALIDSET2 OUT=SORTED_2;
BY DESENDING PREDP_2;
RUN;
PROC PRINT DATA= SORTED_2 (OBS=10);
RUN;

PROC EXPORT DATA=SORTED_2
OUTFILE = 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis\농장등급\sorted_9_result.csv'
dbms=csv replace;
run;
