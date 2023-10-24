/*  SAS Dataset 불러오기 */
libname COW 'C:\Users\Owner\Desktop\윤태준\소\3차 요청자료\230301_Analysis';

data RAW;
set COW.RAW;
run;
