# 01-1 텍스트 전처리

library(rvest)
library(stringr)

# criminal

# 작업용 크롤링 폴더 생성 및 지정
setwd("C:\\Users\\user\\Desktop\\movie")
getwd()

# 네이버 영화 평점 페이지 주소를 변수로 저장: "nhn?code="의 숫자가 영화 제목을 의미
criminal_city = "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=192608&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page="


# 변수의 자료형 정의
review_list = character()
star_list = numeric()
date_list = character()

# 웹페이지의 태그 정보를 읽는 반복문
for(page_url in 1:100){
  url = paste(criminal_city, page_url, sep="")
  content = read_html(url)
  node_1 = html_nodes(content, ".score_reple p")
  node_2 = html_nodes(content, ".score_result_cr .star_score em")
  node_3 = html_nodes(content, ".score_reple em:nth-child(2)")
  review = html_text(node_1)
  star = html_text(node_2)
  date = html_text(node_3)
  date = as.Date(gsub("\\.","-", date))
  review_list = append(review_list, review)
  star_list = append(star_list, star)
  date_list = append(date_list, date)
}

# 데이터 프레임 작성 (영화평, 평점(1-10), 날짜)
df = data.frame(review_list, star_list, date_list)
colnames(df) = c("review","rank","date")
df$review

# 데이터 프레임을 csv 파일로 저장
write.csv(df, "criminal_city.csv", row.names = F)

# 영화평 텍스트만 추출하여 텍스트 파일로 저장
write.table(df$review, file="criminal_city.txt", quote=FALSE, fileEncoding = "UTF-8", col.names = FALSE, row.names = FALSE)

library(KoNLP)

# dplyr 로딩
library(dplyr)
# stringr 로딩
library(stringr)

setwd('C:\\Users\\user\\Desktop\\movie')
criminal <- readLines("criminal_city.txt", encoding = "UTF-8")
head(criminal)
# 텍스트 파일 불러오기
criminal <- criminal %>%
  str_replace_all("[^가-힣]", " ") %>%  # 한글만 남기기
  str_squish()  # 연속된 공백 제거

head(criminal)
criminal <- as_tibble(criminal)
dim(criminal)

## 01-2 토큰화(tokenization)
# 토큰(token): 텍스트의 기본 단위(ex: 단락, 문장, 단어, 형태소)
# 토큰화: 텍스트를 토큰으로 나누는 작업

## tidytext 라이브러리
# 텍스트를 정돈된 데이터(Tidy Data) 형태를 유지하며 분석
# dplyr, ggplot2 패키지와 함께 활용
# 토큰화하기: unnest_tokens()

#install.packages("tidytext")
library(tidytext)
# 문장 기준 토큰화
se_cr <- criminal %>%
  unnest_tokens(input = value,        # 토큰화할 텍스트
                output = word,        # 출력 변수명
                token = "sentences")  # 문장 기준

# 띄어쓰기준 토큰화
wo_cr<- criminal %>%
  unnest_tokens(input = value,
                output = word,
                token = "words")      # 띄어쓰기 기준

# 문자(음절) 기준 토큰화
ch_cr<-criminal %>%
  unnest_tokens(input = value,
                output = word,
                token = "characters")  # 문자 기준

# 단어 단위 토큰화 결과를 word_space 변수에 저장
words_cr <- criminal %>%
  unnest_tokens(input = value,
                output = word,
                token = "words")
ex_cr <- criminal %>%
  unnest_tokens(input = value,
                output = word,
                token = extractNoun)
dim(ex_cr)

# 01-3 단어 빈도 분석

# 값이 동일한 행의 빈도를 측정하여 정렬
cr_frequency <- words_cr %>%
  count(word, sort = T)
# "sort=T" 행 빈도 내림차순으로 정렬 
cr_frequency

# 두 글자 이상의 단어만 남기기
cr02_frequency <- cr_frequency %>%
  filter(str_count(word) > 1)

cr02_frequency

# 고빈도 20개 단어 추출
cr_top20 <- cr02_frequency %>%
  head(20)

cr_top20

# 시각화: 막대 그래프 그리기
# install.packages("ggplot2")
library(ggplot2)

# # 시각화: 막대 그래프 그리기 (2)
ggplot(cr_top20, aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = n), hjust = -0.3) +            # 막대 밖 빈도 표시
  labs(title = "범죄도시2 관람객 리뷰 단어 빈도",  # 그래프 제목
       x = NULL, y = NULL) +                           # 축 이름 삭제
  theme(title = element_text(size = 12))               # 제목 크기


# 구글 폰트 불러오기
#install.packages("jsonlite")
#install.packages("curl")
#install.packages("showtext")
library(showtext)
font_add_google(name = "Nanum Gothic", family = "nanumgothic")
showtext_auto()

## 더 예쁜 워드클라우드 그리기

# 라이브러리 설치
#install.packages('remotes')
library(remotes)
#remotes::install_github("lchiffon/wordcloud2")
library(wordcloud2)

# 라이브러리 설치
wordcloud2(data=cr02_frequency, fontFamily = '나눔바른고딕')


## 제4장. 텍스트의 감정 분석 (1/2)

## 04-1. 감정 사전 활용하기

# 감정 사전 불러오기
install.packages("readr")
library(readr)

# 감성사전 살펴보기
library(dplyr)

positive <- readLines("pol_pos_word.txt", encoding = "UTF-8")
positive = positive[-1]
head(positive)

negative <- readLines("pol_neg_word.txt", encoding = "UTF-8")
negative = negative[-1]
head(negative)

# 사용자 정의 함수: 긍부정어 비교
sentimental = function(sentences, positive, negative){
  scores = laply(sentences, function(sentence, positive, negative) {
    sentence = gsub('[[:punct:]]', '', sentence) # 문장부호 제거
    sentence = gsub('[[:cntrl:]]', '', sentence) # 특수문자 제거
    sentence = gsub('\\d+', '', sentence)        # 숫자 제거
    word.list = str_split(sentence, '\\s+')      # 공백 기준 단어 생성 -> \\s+ : 공백 정규식, +(1개 이상)
    words = unlist(word.list)       # unlist() : list를 vector 객체로 구조변경
    pos.matches = match(words, positive) # words의 단어를 positive에서 matching
    neg.matches = match(words, negative)
    pos.matches = !is.na(pos.matches)            # NA 제거, 위치(숫자)만 추출
    neg.matches = !is.na(neg.matches)
    score = sum(pos.matches) - sum(neg.matches)  # 긍정 - 부정   
    return(score)
  }, positive, negative)
  scores.df = data.frame(score=scores, text=sentences)
  return(scores.df)
}

# 함수에 의하여 긍부정을 분석한 결과를 result_cr 변수에 저장
# 긍정은 1점이상 0점은 중립 부정은 0점 아래로 설정하기 위해 color에 긍정을 blue 중립은 green 부정은 red로 설정

# '영화평 텍스트' 내용 분석
result_cr = sentimental(criminal, positive, negative)
result_cr$color[result_cr$score >=1] = "blue"
result_cr$color[result_cr$score ==0] = "green"
result_cr$color[result_cr$score < 0] = "red"
table(result_cr$color)

# 감성 분석
result_cr$remark[result_cr$score >=1] = "긍정"
result_cr$remark[result_cr$score ==0] = "중립"
result_cr$remark[result_cr$score < 0] = "부정"

circle_cr <-table(result_cr$remark)
circle_cr

label_cr <- paste(names(circle_cr), "\n", circle_cr)
label_cr

# 값(퍼센트) 입력
label_cr <- paste(names(circle_cr), "\n", circle_cr/sum(circle_cr)*100)
pct_cr <- round(circle_cr/sum(circle_cr)*100,2)
label_cr <- paste(names(circle_cr), "\n", pct_cr, "%")
label_cr

# 3D 원그래프 작성 라이브러리 설치 및 적용
# install.packages("plotrix")
library(plotrix)
pie3D(circle_cr, labels=label_cr)
pie3D(circle_cr, labels=label_cr,
      col=c("blue","red","green"), labelcex=1.0, explode=0.2, theta =1,
      main='범죄도시2 감성분석 결과 piechart')
pie3D

