/*  SAS Dataset 불러오기 */
libname COW 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis\농장등급';

data RAW; set COW.RAW; run;
data TESTSET; set COW.TESTSET; run;
data VALIDSET; set COW.VALIDSET; run;


data TESTSET; 
set TESTSET; 
run;

proc freq data= testset;
tables target1;
run;

/* Logistic regression for 7, 8, 9 - AUC 0.638 */
proc logistic data=TESTSET plots=ROC;
class  FARM_LEVEL GENDER (ref='거세');
model TARGET1 (event='1') =  GENDER FARM_LEVEL SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C
                                               A_S_M_W A_S_M_I A_S_F_M A_S_T_M A_S_C 
											   B_S_M_W B_S_M_I B_S_F_M B_S_T_M B_S_C;
run;

* GENDER S_M_W S_M_I S_F_M S_C A_S_M_W A_S_M_I A_S_F_M A_S_T_M A_S_C B_S_M_W B_S_M_I B_S_F_M B_S_T_M B_S_C 제외 (p-value 0.2 기준) - AUC 0.628 - 최종 모형 확인 ;
proc logistic data=TESTSET plots=ROC outest=betas;
class FARM_LEVEL GENDER (ref='거세');
model TARGET1 (event='1') = FARM_LEVEL SL_M S_W
                                               S_T_M
                                               ;
run;

/* Logistic regression for 9 -AUC 0.623*/
proc logistic data=TESTSET plots=ROC;
class FARM_LEVEL GENDER (ref='거세');
model TARGET2 (event='1') =  GENDER FARM_LEVEL SL_M S_W
                                               S_M_W S_M_I S_F_M S_T_M S_C
                                               A_S_M_W A_S_M_I A_S_F_M A_S_T_M A_S_C 
											   B_S_M_W B_S_M_I B_S_F_M B_S_T_M B_S_C/ firth;
run;

* GENDER SL_M S_M_W S_M_I S_F_M A_S_M_W A_S_F_M A_S_T_M B_S_M_W B_S_M_I B_S_F_M B_S_T_M 제외 (p-value 0.2 기준) - AUC 0.625 - 최종 모형 확인 ;
proc logistic data=TESTSET plots=ROC outest=betas_9;
class FARM_LEVEL  GENDER (ref='거세');
model TARGET2 (event='1') =  FARM_LEVEL S_W
                                               S_T_M
                                               A_S_M_I  
											   / firth;
run;





