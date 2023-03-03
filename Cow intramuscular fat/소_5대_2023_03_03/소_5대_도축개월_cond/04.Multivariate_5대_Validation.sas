/*  SAS Dataset 불러오기 */
libname COW 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis_5대\age조절';

data RAW; set COW.RAW; run;
data TESTSET; set COW.TESTSET; run;
data VALIDSET; set COW.VALIDSET; run;


* D_S_C 제외 (p-value 0.2 기준) - AUC 0.695 최종모형확인 ;
proc logistic data=TESTSET plots=ROC outest=betas;
class FARM_LEVEL (ref='3') GENDER (ref='거세');
model TARGET1 (event='1') =  FARM_LEVEL S_W
                                               S_M_W S_M_I S_F_M S_T_M 
                                               A_S_M_I A_S_T_M  
											   B_S_T_M 
											   C_S_M_I 
											   D_S_M_W D_S_M_I;
run;

* GENDER S_M_W  s_m_i	s_f_m	a_s_m_i	a_s_f_m	a_s_c	b_s_m_w	b_s_m_i	b_s_c	c_s_m_w	c_s_m_i	c_s_t_m	c_s_c	d_s_m_w	d_s_m_i	d_s_f_m	d_s_t_m	d_s_c
 제외 (p-value 0.2 기준) - AUC 0.7313 - 최종모형 확인;
proc logistic data=TESTSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세');
model TARGET2 (event='1') =  FARM_LEVEL SL_M S_W
                                               S_T_M S_C
                                               A_S_M_W A_S_T_M 
											   B_S_F_M B_S_T_M 
											   C_S_F_M / firth;
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
PR = -2.2717 + (1.9563 *F1) + (-0.5966*F2) + (0.00652*S_W) 
		+ (-0.00226*S_M_W) + (0.011*S_M_I) + (-0.0215*S_F_M) + (0.1411*S_T_M) 
		+ (-0.007*A_S_M_I) + (0.097*A_S_T_M)
		+ (-0.0478*B_S_T_M) 
		+ (-0.00688*C_S_M_I) 
		+ (0.00187*D_S_M_W) + (-0.00985*D_S_M_I);

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
PR_2 = -6.4502 + (1.321 *F1) + (-0.1662*F2) + (0.082*SL_M) + (0.00647*S_W) 
			+ (0.17*S_T_M) + (-0.1051*S_C) + (-0.00318*A_S_M_W) + (0.1709*A_S_T_M)
			+ (-0.0247*B_S_F_M) + (-0.0909*B_S_T_M) + (-0.0337*C_S_F_M);

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
%diag(TARGET1, PRED5, VALIDSET2);
TITLE2 'CUT-OFF 0.6'; /* sensitiviy 0.04 */
%diag(TARGET1, PRED6, VALIDSET2);
TITLE2 'CUT-OFF 0.7'; /* sensitiviy 0.01 */
%diag(TARGET1, PRED7, VALIDSET2);
TITLE2 'CUT-OFF 0.8'; /* sensitiviy 0 */
%diag(TARGET1, PRED8, VALIDSET2);
TITLE2 'CUT-OFF 0.9'; /* sensitiviy 0 */
%diag(TARGET1, PRED9, VALIDSET2);

/* target_789, proba_ desending */
proc sort data=VALIDSET1 OUT=SORTED_1;
BY DESENDING PREDP;
RUN;
PROC PRINT DATA= SORTED_1 (OBS=10);
RUN;

PROC EXPORT DATA=SORTED_1
OUTFILE = 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis_5대\age조절\sorted_789_result.xlsx'
dbms=xlsx replace;
run;

/* 정렬하고 target이랑 cut-off 0.5 to 0.9 by 0.1 정리한 것 */
proc sort data=VALIDSET2 OUT=SORTED_2;
BY DESENDING PREDP_2;
RUN;
PROC PRINT DATA= SORTED_2 (OBS=10);
RUN;

PROC EXPORT DATA=SORTED_2
OUTFILE = 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis_5대\age조절\sorted_9_result.csv'
dbms=csv replace;
run;

/* validation predict 정리 */
PROC EXPORT DATA=VALIDSET_1
OUTFILE = 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis_5대\age조절\VALID_1.xlsx'
dbms=xlsx replace;
run;

PROC EXPORT DATA=VALIDSET_2
OUTFILE = 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis_5대\age조절\VALID_2.xlsx'
dbms=xlsx replace;
run;
