setwd("C:\\Users\\user\\Desktop\\통계학\\머신러닝")
air.tr<-read.csv("train.csv",stringsAsFactors = TRUE)
tr1<-read.csv("train.csv",stringsAsFactors = TRUE)
str(air.tr);str(tr1)

#데이터 탐색
head(air.tr)
summary(air.tr)

#결측치 모두 0으로 처리
sum(is.na(air.tr)); sum(is.na(tr1))
abcd<-ifelse(is.na(tr1$Arrival.Delay.in.Minutes),0,tr1$Arrival.Delay.in.Minutes)
tr1$Arrival.Delay.in.Minutes<-abcd
air.tr$Arrival.Delay.in.Minutes<-abcd
sum(is.na(air.tr)); sum(is.na(tr1))
# 중앙값으로 처리해줬음
#id와 x 없애기
tr1<-tr1[c(-1,-2)]

# 회귀분석을 위한 수치형 변환
air.tr$sat.dum<-ifelse(air.tr$satisfaction=="satisfied",1,2)
air.tr$gen.dum<-ifelse(air.tr$Gender=="Male",1,2)
air.tr$ct.dum<-ifelse(air.tr$Customer.Type=="Loyal Customer",1,2)
air.tr$tot.dum<-ifelse(air.tr$Type.of.Travel=="Personal Travel",1,2)
air.tr$cl.dum<-ifelse(air.tr$Class=="Business",1,ifelse(air.tr$Class=="Eco",2,3))
str(air.tr)
cr.ar<-air.tr[,-c(1,2,3,4,6,7,25)]
str(cr.ar)

#그룹별로 시각화하기 위한 라벨 작업
cr1<-cr.ar[ ,c(1,2,17,18,19,20,21,22,23)] #
cr2<-cr.ar[ ,-c(1,2,17,18,19,20,21,22,23)] #만족도0,1~5
cra<-cr.ar
#만족도 뺀 나머지 
gef <- function(i)

{
  
  if( i == 1 ) 
    
  {
    
    return( "남" )
    
  }
  
  return( "여" )
  
}

agef <- function(i)   #나이 최소7 최대85
  
{
  
  if( 0<i&i<= 9 ) 
    
  {
    
    return( "10대 미만" )
    
  }
  
  else if( 10<=i&i<=19 )
    
  {
    
    return( "10대" )
    
  }
  
  if( 20<i&i<= 29 ) 
    
  {
    
    return( "20대 " )
    
  }
  
  else if( 30<=i&i<=39 )
    
  {
    
    return( "30대 " ) 
    
  }
  
  else if( 40<=i&i<=49 )
    
  {
    
    return( "40대 " ) 
    
  }
  
  else if( 50<=i&i<=59 )
    
  {
    
    return( "50대 " ) 
    
  }
  
  else if( 60<=i&i<=69 )
    
  { 
    
    return( "60대 " ) 
    
  }
  
  else if( 70<=i&i<=79 )
    
  {
    
    return( "70대 " ) 
    
  }
  
  return( "80대 이상" )
  
}

custf <- function(i)
  
{
  
  if( i == 1 ) 
    
  {
    
    return( "Loyal customer" )
    
  }
  
  
  
  return( "disLoyal Customer" )
  
}

totf <- function(i)
  
{
  
  if( i == 1 ) 
    
  {
    
    return( "Personal Travel" )
    
  }
  
  
  
  return( "Business Travel" )
  
}

classf <- function(i)
  
{
  
  if( i == 1 ) 
    
  {
    
    return( "Business" )
    
  }
  
  if( i == 2 ) 
    
  {
    
    return( "Eco" )
    
  }
  
  
  
  return( "Eco Plus" )
  
}

satf <- function(i)
  
{
  
  if( i == 1 ) 
    
  {
    
    return( "satisfaction" )
    
  }
  
  return( "neutral or dissatisfied" )
  
}





cra$gen.dum<-sapply(unlist(cr1$gen.dum),gef) #성별

cra$Age<-sapply(unlist(cr1$Age),agef) #나이

cra$ct.dum<-sapply(unlist(cr1$ct.dum),custf) #고객유형

cra$tot.dum<-sapply(unlist(cr1$tot.dum),totf) # 여행유형

cra$cl.dum<-sapply(unlist(cr1$cl.dum),classf) # 클래스

cra$sat.dum<-sapply(unlist(cr1$sat.dum),satf) #만족도



#만족도 조사 라벨 , 0있는거 구분해서 따로 해주기

sa <- function(i)
  
{
  
  if( i == 1 | i==0)
    
  {
    
    return( "매우불만족" )
    
  }
  
  else if( i == 2 )
    
  {
    
    return( "약간불만족" )
    
  }
  
  else if( i == 3 )
    
  {
    
    return( "보통" )
    
  }
  
  else if( i == 4 )
    
  {
    
    return( "약간 만족" )
    
  }  
  
  return( "매우 만족" )
  
}

cra$Inflight.wifi.service<-sapply(unlist(cr2$Inflight.wifi.service),sa)

cra$Departure.Arrival.time.convenient<-sapply(unlist(cr2$Departure.Arrival.time.convenient),sa)

cra$Ease.of.Online.booking<-sapply(unlist(cr2$Ease.of.Online.booking),sa)

cra$Gate.location<-sapply(unlist(cr2$Gate.location),sa)

cra$Food.and.drink<-sapply(unlist(cr2$Food.and.drink),sa)

cra$Online.boarding<-sapply(unlist(cr2$Online.boarding),sa)

cra$Seat.comfort<-sapply(unlist(cr2$Seat.comfort),sa)

cra$Inflight.entertainment<-sapply(unlist(cr2$Inflight.entertainment),sa)

cra$On.board.service<-sapply(unlist(cr2$On.board.service),sa)

cra$Leg.room.service<-sapply(unlist(cr2$Leg.room.service),sa)

cra$Baggage.handling<-sapply(unlist(cr2$Baggage.handling),sa)

cra$Checkin.service<-sapply(unlist(cr2$Checkin.service),sa)

cra$Inflight.service<-sapply(unlist(cr2$Inflight.service),sa)

cra$Cleanliness<-sapply(unlist(cr2$Cleanliness),sa)

str(cra)

cra$sat.dum<-as.factor(cra$sat.dum)


#변수선택, 3개의 방법 다 22개 나옴 그냥 변수 23개 다 쓰기로 결정

library(MASS)

lma<-lm(formula=sat.dum~.,data=cr.ar)

lmab<-step(lma,  direction = "both")

lmaback<-step(lma,  direction = "backward")

lmafor<-step(lma,direction="forward")



#그래프, 연속형에서 라벨 씌워준 cra변수사용
ds <- table(   cra$sat.dum,cra$gen.dum ) #성별에 따른 만족도
barplot( ds , beside = T ,  legend.text = rownames(ds) ,xlab = "gen" , ylab = "satis",col=c("blue","pink") )

sc <- table(   cra$sat.dum, cra$cl.dum ) # 등급에 따른 만족도
barplot( sc , beside = T ,  legend.text = rownames(sc),xlab="class",ylab="satis" ,col=c("green","red"))

st<-table(cra$sat.dum, cra$tot.dum) # 여행 목적에 따른 만족도
barplot(st,beside=T, legend.text=rownames(st),xlab="type of travel",ylab="satis",col=c("purple","yellow"))
sct<-table(cra$sat.dum,cra$ct.dum)
barplot(sct,beside=T, legend.text=rownames(sct),xlab="customer type",ylab="satis",col=c("aquamarine","coral"))
#변수들 끼리의 상관관계 파악
cor.ar<-cor(cr.ar)
cor.ar<-round(cor.ar,digit=2)

#확률계산
table(air.tr$sat.dum)
round(prop.table(table(air.tr$sat.dum)),digit=2)

#의사결정트리, train데이터 자체가 10만개이기 때문에 여기에서 개수를 나눠서 분석,, 원래 train과test있는걸로 
#2개 따로 모델을 만들어 분석
library(C50)
dim(tr1)

modtr<-sample(tr1[1:72733,])
modte<-sample(tr1[72534:103904,])

#sample함수를 이용하여 추출했기 때문에 모델의 목표변수의 열을 수시로 체크해야함
names(modtr)

#라벨 열 제거하고 라벨 지정해주기
air_model<-C5.0(modtr[-12],modtr$satisfaction) 

# 분배된 데이터 라벨값 분포 보기
round(prop.table(table(modtr$satisfaction)),2)
round(prop.table(table(modte$satisfaction)),2)

air_model

summary(air_model) #트리 사이즈295개

install.packages("tree")
library(tree)

#모델 테스트셋 적용
tree_eval<-predict(air_model,modte)
library(gmodels)
CrossTable(modte$satisfaction,tree_eval,
           prop.chisq=FALSE, prop.c = FALSE,prop.r = FALSE,
           dnn=c("actual satisfaction","pred satisfaction")) #정확도 96%

# 트리사이즈가 너무 커서 보이지 않음 다른 방법을 통한 시각화
treemod<-tree(satisfaction~.,data=modtr)
summary(treemod)
dev.new()
plot(treemod)
text(treemod)

#반복을 통한 모델 개선
#튜닝모델
install.packages("caret")
library(caret)


#붓스태랩을 통한 방법trial 20, winnow=F, model=Rules best model

tmt<-train(formula=satisfaction~.,data= modtr,method='C5.0')

#예측 실행 표 수정 전 모델
air_pp<-predict(air_model,modte,type="prob")
predict2<-tree_eval
predtable<-as.data.frame(modte$satisfaction) #실제값
predtable$predict<-tree_eval #모델넣은 값
predtable$neutral_or_dissatisfied<-air_pp[,1]
predtable$ssatisfied<-air_pp[,2]
colnames(predtable)<-c("actual"  , "predict","neutral_or_dissatisfied" ,"satisfied" )
head(predtable)
table(predtable$actual,predtable$predict)
CrossTable(predtable$actual,predtable$predict)
confusionMatrix(predtable$actual,predtable$predict,positive="neutral or dissatisfied")

#roc 커브 수정 전
predict2<-predict(air_model,modte,type = 'prob')
real<-modte$satisfaction
library(pROC)
roc<-roc(real,predtable$satisfied)
dev.new()
plot.roc(roc,
         col='blue',   # 선의 색
         print.auc=TRUE,  #auc 출력 
         print.auc.col='red', #auc 색
         print.thres=TRUE, # theshold 출력 
         print.thres.pch=19, #theshold 점 모양
         print.thres.col = "red", #threhold 색
         grid=c(0.2, 0.2), #격자
         cex.lab=1.2,legacy.axes = TRUE) # x,y 레이블 크기

#Rules를 사용해 트리 만들기 156개
c5_options <- C5.0Control(winnow = FALSE, noGlobalPruning = FALSE)
c5_model <- C5.0(satisfaction~., data=modtr,control=c5_options, rules=TRUE,trials=20)
summary(c5_model)
rule_pred<-predict(c5_model,modte)
a<-CrossTable(modte$satisfaction,rule_pred,
              prop.chisq=FALSE, prop.c = FALSE,prop.r = FALSE,
              dnn=c("actual satisfaction","pred satisfaction")) #정확도 96%



#예측 실행 표
pred_pr<-predict(c5_model,modte,type="prob")
predict2<-rule_pred
result<-as.data.frame(modte$satisfaction) #실제값
result$predict<-rule_pred #모델넣은 값
result$neutral_or_dissatisfied<-pred_pr[,1]
result$ssatisfied<-pred_pr[,2]
colnames(result)<-c("actual"  , "predict","neutral_or_dissatisfied" ,"satisfied" )
head(result)
table(result$actual,result$predict)
CrossTable(result$actual,result$predict)
confusionMatrix(result$actual,result$predict,positive="neutral or dissatisfied")

#roc 커브
predict2<-predict(c5_model,modte,type = 'prob')
real<-modte$satisfaction
library(pROC)
roc<-roc(real,result$satisfied)
dev.new()
plot.roc(roc,
         col='blue',   # 선의 색
         print.auc=TRUE,  #auc 출력 
         print.auc.col='red', #auc 색
         print.thres=TRUE, # theshold 출력 
         print.thres.pch=19, #theshold 점 모양
         print.thres.col = "red", #threhold 색
         grid=c(0.2, 0.2), #격자
         cex.lab=1.2,legacy.axes = TRUE) # x,y 레이블 크기
