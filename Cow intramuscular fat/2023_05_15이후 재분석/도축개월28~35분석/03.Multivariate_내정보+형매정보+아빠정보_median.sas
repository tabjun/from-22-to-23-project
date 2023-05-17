/*  SAS Dataset 불러오기 */
/*  SAS Dataset 불러오기 */
libname COW 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230504_Analysis\Analysis_0515\도축개월\Data';

data RAW; set COW.RAW; run;
data TRAINSET; set COW.TRAINSET; run;
data VALIDSET; set COW.VALIDSET; run;

PROC PRINT DATA=TRAINSET(OBS=10); RUN;

data TRAINSET;
set TRAINSET;
run;

/* -----------------------------------------------TARGET1----------------------------------------------------------------*/

/* 등급 7,8,9 event 1로 변경한 것 ==> TARGET1 */
/* 내 정보 + 형매 정보 + 아빠 정보(1. 중앙값으로 ABC나누기한거, kpn_mean_class) */
/* Logistic regression for 7, 8, 9 AUC - 0.69*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세') kpn_median_class(ref='C');
model TARGET1 (event='1')  =  GENDER FARM_LEVEL kpn_median_class SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C ;
run;

/* 내 정보 + 형매 정보 + 아빠 정보(1. 중앙값으로 등급 나누기한거 연속형으로 표시- kpn_median_level) */
/* Logistic regression for 7, 8, 9 AUC - 0.69*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세') ;
model TARGET1 (event='1')  =  GENDER FARM_LEVEL kpn_median_level SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C ;
run;

/* 내 정보 + 형매 정보 + 아빠 정보(1. 중앙값으로 1~9 나누기한거 연속형으로 표시- kpn_median) */
/* Logistic regression for 7, 8, 9 AUC - 0.7*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세');
model TARGET1 (event='1')  =  GENDER FARM_LEVEL kpn_median SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C ;
run;

/* -----------------------------------------------TARGET2----------------------------------------------------------------*/

/* 등급 9 event 1로 변경한 것 ==> TARGET2 */
/* 내 정보 + 형매 정보 + 아빠 정보(1. 중앙값으로 ABC나누기한거, kpn_median_class) */
/* Logistic regression for 9 AUC - 0.72*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세') kpn_median_class(ref='C');
model TARGET2 (event='1')  =  GENDER FARM_LEVEL kpn_median_class SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C ;
run;

/* 내 정보 + 형매 정보 + 아빠 정보(1. 중앙값으로 등급 나누기한거 연속형으로 표시- kpn_median_level) */
/* Logistic regression for 9 AUC - 0.72*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세') ;
model TARGET2 (event='1')  =  GENDER FARM_LEVEL kpn_median_level SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C ;
run;

/* 내 정보 + 형매 정보 + 아빠 정보(1. 중앙값으로 1~9 나누기한거 연속형으로 표시- kpn_median) */
/* Logistic regression for 9 AUC - 0.74*/
proc logistic data=TRAINSET plots=ROC;
class FARM_LEVEL (ref='3') GENDER (ref='거세');
model TARGET2 (event='1')  =  GENDER FARM_LEVEL kpn_median SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C ;
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

if kpn_median_class = 'A' then k1=1;
if kpn_median_class = 'B' then k1=0;
if kpn_median_class = 'C' then k1=0;

if kpn_median_class = 'A' then k2=0;
if kpn_median_class = 'B' then k2=1;
if kpn_median_class = 'C' then k2=0;
run;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(1. 중앙값으로 ABC나누기한 것), TARGET 789 */ 
data VALIDSET_1;
set VALIDSET;

/* target 789 */
PR = -3.4101 + (-0.0487*SEX) + (1.2943*F1) + (-0.3013*F2) + (1.0028*k1) + (-0.046*k2) 
		+ (0.0164*SL_M) + (0.00577*S_W) 
		+ (-0.00216*S_M_W) + (0.00172* S_M_I) +(-0.00881*S_F_M)+ (0.184*S_T_M) + (-0.00421*S_C); 

PREDP = exp(PR)/(1+exp(PR));

if PREDP >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP >= 0.9 then PRED9 =1; else PRED9 = 0;

run;

proc print data=VALIDSET_1 (OBS=10);
run;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(중앙값으로 등급을 1,2,3 나누기한 것), TARGET 789  */
data VALIDSET_2;
set VALIDSET;

/* target 789 */
PR_2 = -5.5497 + (-0.0486*SEX) + (1.2944 *F1) + (-0.3013*F2) + (1.047*KPN_MEDIAN_LEVEL) 
		+ (0.0164*SL_M) + (0.00577*S_W) 	
		+ (-0.00216*S_M_W) + (0.00172* S_M_I) +(-0.00882*S_F_M)+ (0.184*S_T_M) + (-0.00421*S_C); 

PREDP_2 = exp(PR_2)/(1+exp(PR_2));

if PREDP_2 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_2 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_2 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_2 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_2 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_2(OBS=10);
RUN;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(중앙값으로 1~9 나누기한 것), TARGET 7,8,9  */
data VALIDSET_3;
set VALIDSET;

/* target only 9 */
PR_3 =-6.82 + (-0.0636*SEX) + (1.2572 *F1) + (-0.2915*F2) + (0.5836*KPN_MEDIAN) 
		+ (0.0219*SL_M) + (0.00569*S_W) 	
		+ (-0.00229*S_M_W) + (0.00144* S_M_I) +(-0.00827*S_F_M)+ (0.1826*S_T_M) + (0.00869*S_C); 

PREDP_3 = exp(PR_3)/(1+exp(PR_3));

if PREDP_3 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_3 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_3 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_3 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_3 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_3(OBS=10);
RUN;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(중앙값으로 ABC 나누기한 것), TARGET 9  */
data VALIDSET_4;
set VALIDSET;

/* target 9 */
PR_4 = -5.9674+ (0.0916*SEX) + (1.1951 *F1) + (-0.1693*F2) + (1.1442*k1) + (-0.0717*k2) 
		+ (0.0296*SL_M) + (0.00644*S_W) 
		+ (-0.00231*S_M_W) + (0.00455* S_M_I) +(-0.0126*S_F_M)+ (0.2097*S_T_M) + (-0.0582*S_C); 

PREDP_4 = exp(PR_4)/(1+exp(PR_4));

if PREDP_4 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_4 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_4 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_4 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_4 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_4(OBS=10);
RUN;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(중앙값으로 1,2,3 나누기한 것), TARGET 9  */
data VALIDSET_5;
set VALIDSET;

/* target 9 */
PR_5 =  -8.4688 + (0.0917*SEX) + (1.195*F1) + (-0.1693*F2) + (1.215*KPN_MEDIAN_LEVEL) 
		+ (0.0296*SL_M) + (0.00644*S_W) 	
		+ (-0.00231*S_M_W) + (0.00455* S_M_I) +(-0.0126*S_F_M)+ (0.2097*S_T_M) + (-0.0582*S_C); 

PREDP_5 = exp(PR_5)/(1+exp(PR_5));

if PREDP_5 >= 0.5 then PRED5 =1; else PRED5 = 0;
if PREDP_5 >= 0.6 then PRED6 =1; else PRED6 = 0;
if PREDP_5 >= 0.7 then PRED7 =1; else PRED7 = 0;
if PREDP_5 >= 0.8 then PRED8 =1; else PRED8 = 0;
if PREDP_5 >= 0.9 then PRED9 =1; else PRED9 = 0;
RUN;

PROC PRINT DATA= VALIDSET_5(OBS=10);
RUN;

/* 내 정보 + 농가 정보 + 형매 정보 + 아빠 정보(중앙값으로 1,2,3 나누기한 것), TARGET 9  */
data VALIDSET_6;
set VALIDSET;

/* target 9 */
PR_6 =  -10.5389+ (0.0754*SEX) + (1.1475*F1) + (-0.1573*F2) + (0.787*KPN_MEDIAN) 
		+ (0.0345*SL_M) + (0.00629*S_W) 	
		+ (-0.00243*S_M_W) + (0.00431* S_M_I) +(-0.012*S_F_M)+ (0.2067*S_T_M) + (-0.0469*S_C); 

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
