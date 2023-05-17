/*  SAS Dataset 불러오기 */
/*  SAS Dataset 불러오기 */
libname COW 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230504_Analysis\Analysis_0515\DATASET';

data RAW; set COW.RAW; run;
data TRAINSET; set COW.TRAINSET; run;
data VALIDSET; set COW.VALIDSET; run;

PROC PRINT DATA=TRAINSET(OBS=10); RUN;

data TRAINSET;
set TRAINSET;
run;

/* -----------------------------------------------TARGET1----------------------------------------------------------------*/

/* 등급 7,8,9 event 1로 변경한 것 ==> TARGET1 */
/* 내 정보 + 형매 정보 + 아빠 정보(1. 평균값으로 ABC나누기한거, kpn_mean_class) */
/* Logistic regression for 7, 8, 9 AUC - 0.72*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세') kpn_mean_class(ref='C');
model TARGET1 (event='1')  =  GENDER FARM_LEVEL kpn_mean_class SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C ;
run;

/* 내 정보 + 형매 정보 + 아빠 정보(1. 평균값으로 등급 나누기한거 연속형으로 표시- kpn_mean_level) */
/* Logistic regression for 7, 8, 9 AUC - 0.72*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세') ;
model TARGET1 (event='1')  =  GENDER FARM_LEVEL kpn_mean_level SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C ;
run;

/* 내 정보 + 형매 정보 + 아빠 정보(1. 평균값으로 1~9 나누기한거 연속형으로 표시- kpn_mean) */
/* Logistic regression for 7, 8, 9 AUC - 0.74*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세');
model TARGET1 (event='1')  =  GENDER FARM_LEVEL kpn_mean SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C ;
run;

/* -----------------------------------------------TARGET2----------------------------------------------------------------*/

/* 등급 9 event 1로 변경한 것 ==> TARGET2 */
/* 내 정보 + 형매 정보 + 아빠 정보(1. 평균값으로 ABC나누기한거, kpn_median_class) */
/* Logistic regression for 9 AUC - 0.75*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세') kpn_mean_class(ref='C');
model TARGET2 (event='1')  =  GENDER FARM_LEVEL kpn_mean_class SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C / firth;
run;

/* 내 정보 + 형매 정보 + 아빠 정보(1. 평균값으로 등급 나누기한거 연속형으로 표시- kpn_median_level) */
/* Logistic regression for 9 AUC - 0.75*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세') ;
model TARGET2 (event='1')  =  GENDER FARM_LEVEL kpn_mean_level SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C /firth;
run;

/* 내 정보 + 형매 정보 + 아빠 정보(1. 평균값으로 1~9 나누기한거 연속형으로 표시- kpn_median) */
/* Logistic regression for 9 AUC - 0.77*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세');
model TARGET2 (event='1')  =  GENDER FARM_LEVEL kpn_mean SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C /firth;
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

if kpn_mean_class = 'A' then k1=1;
if kpn_mean_class = 'B' then k1=0;
if kpn_mean_class = 'C' then k1=0;

if kpn_mean_class = 'A' then k2=0;
if kpn_mean_class = 'B' then k2=1;
if kpn_mean_class = 'C' then k2=0;
run;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(1. 평균값으로 ABC나누기한 것), TARGET 789 */ 
data VALIDSET_1;
set VALIDSET;

/* target 789 */
PR = -3.4022 + (-0.0497*SEX) + (1.4123 *F1) + (-0.3264*F2) + (1.2245*k1) + (0.3061*k2) 
		+ (-0.00523*SL_M) + (0.00672*S_W) 
		+ (-0.00206*S_M_W) + (0.000356* S_M_I) +(-0.00784*S_F_M)+ (0.1789*S_T_M) + (-0.00705*S_C); 

PREDP = exp(PR)/(1+exp(PR));

if PREDP >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP >= 0.9 then PRED9 =1; else PRED9 = 0;

run;

proc print data=VALIDSET_1 (OBS=10);
run;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(평균값으로 등급을 1,2,3 나누기한 것), TARGET 789  */
data VALIDSET_2;
set VALIDSET;

/* target 789 */
PR_2 = -4.9469 + (-0.0498*SEX) + (1.4125 *F1) + (-0.3265*F2) + (0.925*KPN_MEAN_LEVEL) 
		+ (-0.00523*SL_M) + (0.00672*S_W) 	
		+ (-0.00206*S_M_W) + (0.000358* S_M_I) +(-0.00783*S_F_M)+ (0.1789*S_T_M) + (-0.00707*S_C); 

PREDP_2 = exp(PR_2)/(1+exp(PR_2));

if PREDP_2 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_2 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_2 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_2 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_2 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_2(OBS=10);
RUN;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(평균값으로 1~9 나누기한 것), TARGET 7,8,9  */
data VALIDSET_3;
set VALIDSET;

/* target only 9 */
PR_3 =-6.2728+ (-0.0673*SEX) + (1.3871 *F1) + (-0.3238*F2) + (0.5892*KPN_MEAN) 
		+ (-0.00323*SL_M) + (0.00663*S_W) 	
		+ (-0.00223*S_M_W) + (0.00009* S_M_I) +(-0.00716*S_F_M)+ (0.1776*S_T_M) + (0.00837*S_C); 

PREDP_3 = exp(PR_3)/(1+exp(PR_3));

if PREDP_3 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_3 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_3 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_3 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_3 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_3(OBS=10);
RUN;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(평균값으로 ABC 나누기한 것), TARGET 9  */
data VALIDSET_4;
set VALIDSET;

/* target 9 */
PR_4 = -5.4948+ (0.035*SEX) + (1.2836 *F1) + (-0.1614*F2) + (1.5051*k1) + (0.3718*k2) 
		+ (-0.00778*SL_M) + (0.00718*S_W) 
		+ (-0.00253*S_M_W) + (0.00414* S_M_I) +(-0.00983*S_F_M)+ (0.1968*S_T_M) + (-0.0523*S_C); 

PREDP_4 = exp(PR_4)/(1+exp(PR_4));

if PREDP_4 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_4 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_4 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_4 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_4 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_4(OBS=10);
RUN;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(평균값으로 1,2,3 나누기한 것), TARGET 9  */
data VALIDSET_5;
set VALIDSET;

/* target 9 */
PR_5 =  -7.3955+ (0.035*SEX) + (1.2842 *F1) + (-0.1616*F2) + (1.1361*KPN_MEAN_LEVEL) 
		+ (-0.00778*SL_M) + (0.00719*S_W) 	
		+ (-0.00253*S_M_W) + (0.00414* S_M_I) +(-0.00982*S_F_M)+ (0.1968*S_T_M) + (-0.0523*S_C); 

PREDP_5 = exp(PR_5)/(1+exp(PR_5));

if PREDP_5 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_5 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_5 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_5 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_5 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_5(OBS=10);
RUN;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(평균값으로 1,2,3 나누기한 것), TARGET 9  */
data VALIDSET_6;
set VALIDSET;

/* target 9 */
PR_6 =  -9.1586+ (0.0174*SEX) + (1.2446 *F1) + (-0.1554*F2) + (0.7426*KPN_MEAN) 
		+ (-0.00547*SL_M) + (0.00703*S_W) 	
		+ (-0.00269*S_M_W) + (0.00394* S_M_I) +(-0.00919*S_F_M)+ (0.1938*S_T_M) + (-0.0374*S_C); 

PREDP_6 = exp(PR_6)/(1+exp(PR_6));

if PREDP_6 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_6 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_6 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_6 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_6 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_6(OBS=10);
RUN;


/* Performance Comparisons */
data VALIDSET1;
set VALIDSET_1;
keep ID TARGET1 PREDP PRED5 PRED6 PRED7 PRED8 PRED9;
run;

data VALIDSET2;
set VALIDSET_2;
keep ID TARGET1 PREDP_2 PRED5 PRED6 PRED7 PRED8 PRED9;
run;

data VALIDSET3;
set VALIDSET_3;
keep ID TARGET1 PREDP_3 PRED5 PRED6 PRED7 PRED8 PRED9;
run;

data VALIDSET4;
set VALIDSET_4;
keep ID TARGET2 PREDP_4 PRED5 PRED6 PRED7 PRED8 PRED9;
run;

data VALIDSET5;
set VALIDSET_5;
keep ID TARGET2 PREDP_5 PRED5 PRED6 PRED7 PRED8 PRED9;
run;

data VALIDSET6;
set VALIDSET_6;
keep ID TARGET2 PREDP_6 PRED5 PRED6 PRED7 PRED8 PRED9;
run;

/* Using Macro */
/* TARGET 789 */
TITLE1 'TARGET 789, KPN_MEAN_CLASS';
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

TITLE1 'TARGET 789, KPN_MEAN_LEVEL';
TITLE2 'CUT-OFF 0.5'; /* sensitiviy 51.87 */
%diag(TARGET1, PRED5, VALIDSET2);
TITLE2 'CUT-OFF 0.6'; /* sensitivity 25.08 */
%diag(TARGET1, PRED6, VALIDSET2);
TITLE2 'CUT-OFF 0.7'; /* sensitivity 7.96 */
%diag(TARGET1, PRED7, VALIDSET2);
TITLE2 'CUT-OFF 0.8'; /* sensitivity 2.44 */
%diag(TARGET1, PRED8, VALIDSET2);
TITLE2 'CUT-OFF 0.9'; /* sensitivity 0.51 */
%diag(TARGET1, PRED9, VALIDSET2);

TITLE1 'TARGET 789, KPN_MEAN';
TITLE2 'CUT-OFF 0.5'; /* sensitiviy 0.14 */
%diag(TARGET1, PRED5, VALIDSET3);
TITLE2 'CUT-OFF 0.6'; /* sensitiviy 0.04 */
%diag(TARGET1, PRED6, VALIDSET3);
TITLE2 'CUT-OFF 0.7'; /* sensitiviy 0.01 */
%diag(TARGET1, PRED7, VALIDSET3);
TITLE2 'CUT-OFF 0.8'; /* sensitiviy 0 */
%diag(TARGET1, PRED8, VALIDSET3);
TITLE2 'CUT-OFF 0.9'; /* sensitiviy 0 */
%diag(TARGET1, PRED9, VALIDSET3);


/* TARGET ONLY 9 */
TITLE1 'TARGET ONLY 9';
TITLE2 'CUT-OFF 0.5'; /* sensitiviy 0.14 */
%diag(TARGET2, PRED5, VALIDSET4);
TITLE2 'CUT-OFF 0.6'; /* sensitiviy 0.04 */
%diag(TARGET2, PRED6, VALIDSET4);
TITLE2 'CUT-OFF 0.7'; /* sensitiviy 0.01 */
%diag(TARGET2, PRED7, VALIDSET4);
TITLE2 'CUT-OFF 0.8'; /* sensitiviy 0 */
%diag(TARGET2, PRED8, VALIDSET4);
TITLE2 'CUT-OFF 0.9'; /* sensitiviy 0 */
%diag(TARGET2, PRED9, VALIDSET4);

TITLE1 'TARGET ONLY 9';
TITLE2 'CUT-OFF 0.5'; /* sensitiviy 0.14 */
%diag(TARGET2, PRED5, VALIDSET5);
TITLE2 'CUT-OFF 0.6'; /* sensitiviy 0.04 */
%diag(TARGET2, PRED6, VALIDSET5);
TITLE2 'CUT-OFF 0.7'; /* sensitiviy 0.01 */
%diag(TARGET2, PRED7, VALIDSET5);
TITLE2 'CUT-OFF 0.8'; /* sensitiviy 0 */
%diag(TARGET2, PRED8, VALIDSET5);
TITLE2 'CUT-OFF 0.9'; /* sensitiviy 0 */
%diag(TARGET2, PRED9, VALIDSET5);

TITLE1 'TARGET ONLY 9';
TITLE2 'CUT-OFF 0.5'; /* sensitiviy 0.14 */
%diag(TARGET2, PRED5, VALIDSET6);
TITLE2 'CUT-OFF 0.6'; /* sensitiviy 0.04 */
%diag(TARGET2, PRED6, VALIDSET6);
TITLE2 'CUT-OFF 0.7'; /* sensitiviy 0.01 */
%diag(TARGET2, PRED7, VALIDSET6);
TITLE2 'CUT-OFF 0.8'; /* sensitiviy 0 */
%diag(TARGET2, PRED8, VALIDSET6);
TITLE2 'CUT-OFF 0.9'; /* sensitiviy 0 */
%diag(TARGET2, PRED9, VALIDSET6);


/* file export to xlsx */
/* target_789, proba_ desending */
proc sort data=VALIDSET1 OUT=SORTED_1;
BY DESENDING PREDP;
RUN;
PROC PRINT DATA= SORTED_1 (OBS=10);
RUN;

PROC EXPORT DATA=SORTED_1
OUTFILE = 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230504_Analysis\결과 정리\엄마정보없이_mean등급_789.xlsx'
dbms=xlsx replace;
run;
