libname GEE "C:\Users\rnjst\OneDrive - 계명대학교\★통계분석\영상의학과_이두영선생님 (GEE)";

proc import out=GEE.Train file="C:\Users\rnjst\OneDrive - 계명대학교\★통계분석\영상의학과_이두영선생님 (GEE)\DATA.xlsx" ;
sheet = 'Training'; run;
proc import out=GEE.Test file="C:\Users\rnjst\OneDrive - 계명대학교\★통계분석\영상의학과_이두영선생님 (GEE)\DATA.xlsx" ;
sheet = 'Test'; run;

data Train; set GEE.Train; run;
data Test; set GEE.Test; run;




/**** List form data set 만들기 ****/
/*
proc freq data=Train; 
table CT * Pathology / out = Train_CT;
run;
proc freq data=Train; 
table Radiomics * Pathology / out = Train_Radiomics; 
run;

proc freq data=Test; 
table (CT) * Pathology / out = Test_CT; 
run;
proc freq data=Test; 
table (Radiomics) * Pathology / out = Test_Radiomics; 
run;
*/



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


*----GEE(diff);
* link=normal, bin 결과 같음;
%macro GEE_diff_sensitivity(dataset, id, gs, gs_m, method, result, desc);
ods select Type3 LSMeans Diffs;
proc genmod data=&dataset &desc;
	class &id &method;
	model &result=&method/dist=normal link = identity type3 wald;
	repeated subject=&id/corrw type=indep;
	lsmeans &method/diff ;
	where &gs=&gs_m ;
run;
%mend GEE_diff_sensitivity ;

%macro GEE_diff_specificity(dataset, id, gs, gs_b, method, result, desc);
ods select Type3  LSMeans Diffs;
proc genmod data=&dataset &desc;
	class &id &method;
	model &result = &method/dist=normal link=identity type3 wald;
	repeated subject=&id/corrw type=indep;
	lsmeans &method/diff;
	where &gs=&gs_b;
run;
%mend GEE_diff_specificity ;

%macro GEE_diff_accuracy(dataset, id, gs, gs_m, gs_b, method, result, result_m, result_b);
ods select Type3  LSMeans Diffs;
data acc; set &dataset;
	if &result = . or &gs = . then same = .;
	else if &result=&result_m and &gs=&gs_m then same=1;
	else if &result=&result_b and &gs=&gs_b then same=1;
	else if &result=&result_m and &gs=&gs_b then same=0;
	else if &result=&result_b and &gs=&gs_m then same=0;
run ;
proc genmod data = acc desc;
	class &id &method;
	model same=&method/dist=normal link=identity type3 wald;
	repeated subject=&id/corrw type=indep;
	lsmeans &method/diff;
run;
%mend GEE_diff_accuracy ;

%macro GEE_diff_PPV(dataset, id, result, result_m, method, gs, desc);
ods select Type3  LSMeans Diffs;
proc genmod data=&dataset &desc;
	class &id &method;
	model &gs=&method/dist=normal link=identity type3 wald;
	repeated subject=&id/corrw type=indep;
	lsmeans &method/diff;
	where &result=&result_m;
run;
%mend GEE_diff_PPV ;

%macro GEE_diff_NPV(dataset, id, result, result_b, method, gs, desc);
ods select Type3  LSMeans Diffs;
proc genmod data=&dataset &desc;
	class &id &method;
	model &gs=&method/dist=normal link=identity type3 wald;
	repeated subject=&id/corrw type=indep;
	lsmeans &method/diff;
	where &result=&result_b;
run;
%mend GEE_diff_NPV ;







/* Analysis Part */

%diag(Pathology, CT, Train);
%diag(Pathology, Radiomics, Train);
%diag(Pathology, CT, Test);
%diag(Pathology, Radiomics, Test);







* CT training set;
data Train; set Train; no=_N_; run;
data CT_Train; set Train;
	array a{*} CT Radiomics;
	do method = 1 to 2;
	result = a{method};
	output;  end;
run;

%GEE_diff_sensitivity(dataset= CT_Train, id = no, gs = Pathology, gs_m = 1, method=method, result=result, desc=desc);
%GEE_diff_specificity(dataset=CT_Train, id=no, gs=Pathology, gs_b=0, method=method, result=result, desc=);
%GEE_diff_accuracy(dataset=CT_Train, id=no, gs=Pathology, gs_m=1, gs_b=0, method=method, result=result, result_m=1, result_b=0);
%GEE_diff_PPV (dataset=CT_Train, id=no, result=result, result_m=1, method=method, gs=Pathology, desc=desc);
%GEE_diff_NPV (dataset=CT_Train, id=no, result=result, result_b=0, method=method, gs=Pathology, desc=);


data Test; set Test; no=_N_; run;
data CT_Test; set Test;
	array a{*} CT Radiomics;
	do method = 1 to 2;
	result = a{method};
	output;  end;
run;

%GEE_diff_sensitivity(dataset= CT_Test, id = no, gs = Pathology, gs_m = 1, method=method, result=result, desc=desc);
%GEE_diff_specificity(dataset=CT_Test, id=no, gs=Pathology, gs_b=0, method=method, result=result, desc=);
%GEE_diff_accuracy(dataset=CT_Test, id=no, gs=Pathology, gs_m=1, gs_b=0, method=method, result=result, result_m=1, result_b=0);
%GEE_diff_PPV (dataset=CT_Test, id=no, result=result, result_m=1, method=method, gs=Pathology, desc=desc);
%GEE_diff_NPV (dataset=CT_Test, id=no, result=result, result_b=0, method=method, gs=Pathology, desc=);






/* 빈도수 체크 */
proc freq data= Train_CT; 
weight Count;
table CT * Pathology;
run;
proc freq data= Train_radiomics;
weight Count;
table Radiomics * Pathology;
run;

proc freq data= Test_CT; 
weight Count;
table CT * Pathology;
run;
proc freq data= Test_Radiomics;
weight Count;
table Radiomics * Pathology;
run;

/* proc genmod 공부는 해야할듯. */

proc freq data=CT_Train; 
table Result; run;

ods select Type3 LSMeans Diffs;
proc genmod data=CT_Train desc;
	class no method;
	model result=method/dist=normal link = identity type3 wald;
	repeated subject=no/corrw type=indep;
	lsmeans method/diff ;
	where Pathology=1 ;
run;





