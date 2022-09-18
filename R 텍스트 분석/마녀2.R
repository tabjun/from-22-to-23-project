# 01-1 텍스트 전처리

# dplyr 로딩
library(dplyr)
# stringr 로딩
library(stringr)

setwd('C:\\Users\\user\\Desktop\\movie')
witch <- readLines("witch_review.txt", encoding = "UTF-8")
head(witch)

txt <- "치킨은!! 맛있다. xyz 정말 맛있다!@#"
txt

str_replace_all(string = witch, pattern = "[^가-힣]", replacement = " ")

# 파라미터명 입력
# str_replace_all(string = txt, pattern = "[^가-힣]", replacement = " ")

# 파라미터명 생략
# str_replace_all(txt, "[^가-힣]", " ")


# 텍스트 파일 불러오기
witc <- witch %>%
  str_replace_all("[^가-힣]", " ") %>%
  str_squish()

head(witc)
witc <- as_tibble(witc)
witc

## tibble 자료형 (dataframe과 유사)
# 한 행에 한 단락이 들어있음
# 긴 문장은 Console 창에서 보기 편할 만큼 일부만 출력
# 행과 열의 수를 알 수 있음
# 변수의 자료형을 알 수 있음

## 전처리 작업 요약
# witc <- witch %>%
#  str_replace_all("[^가-힣]", " ") %>%  # 한글만 남기기
#  str_squish() %>%                      # 연속된 공백 제거
#  as_tibble()                           # tibble로 변환

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
se_wi <- witc %>%
  unnest_tokens(input = value,        # 토큰화할 텍스트
                output = word,        # 출력 변수명
                token = "sentences")  # 문장 기준

# 띄어쓰기준 토큰화
wo_wi<- witc %>%
  unnest_tokens(input = value,
                output = word,
                token = "words")      # 띄어쓰기 기준

# 문자(음절) 기준 토큰화
ch_wi<-witc %>%
  unnest_tokens(input = value,
                output = word,
                token = "characters")  # 문자 기준

# 단어 단위 토큰화 결과를 word_space 변수에 저장
words_wi <- witc %>%
  unnest_tokens(input = value,
                output = word,
                token = "words")
words_wi;wo_wi

# 01-3 단어 빈도 분석

# 값이 동일한 행의 빈도를 측정하여 정렬
word_frequency <- words_wi %>%
  count(word, sort = T)
# "sort=T" 행 빈도 내림차순으로 정렬 
word_frequency

# str_count: 단어의 길이(음절 개수) 반환
str_count("배")
str_count("사과")

# 두 글자 이상의 단어만 남기기
word02_frequency <- word_frequency %>%
 filter(str_count(word) > 1)

word02_frequency

# 중간 변수 없이 한 번에 전처리
# 빈도 내림차순 정렬 후 두 글자 이상 단어 남기기
word02_frequency <- words_wi %>%
 count(word, sort = T) %>%
 filter(str_count(word) > 1)

# 고빈도 20개 단어 추출
top20 <- word02_frequency %>%
 head(20)

top20

# 시각화: 막대 그래프 그리기
# install.packages("ggplot2")
library(ggplot2)

ggplot(top20, aes(x = reorder(word, n), y = n)) +  # 단어 빈도순 정렬
 geom_col() +
 coord_flip()                                      # 회전

# # 시각화: 막대 그래프 그리기 (2)
ggplot(top20, aes(x = reorder(word, n), y = n)) +
 geom_col() +
 coord_flip() +
 geom_text(aes(label = n), hjust = -0.3) +            # 막대 밖 빈도 표시
 labs(title = "문재인 대통령 출마 연설문 단어 빈도",  # 그래프 제목
      x = NULL, y = NULL) +                           # 축 이름 삭제
 theme(title = element_text(size = 12))               # 제목 크기

# 시각화: 워드클라우드 그리기
#install.packages("ggwordcloud")
library(ggwordcloud)

ggplot(word02_frequency, aes(label = word, size = n)) +
  geom_text_wordcloud(seed = 1234) +
  scale_radius(limits = c(3, NA),     # 최소, 최대 단어 빈도
              range = c(3, 30))      # 최소, 최대 글자 크기

# 시각화: 워드클라우드 그리기 (2)
ggplot(word02_frequency, 
       aes(label = word, 
           size = n, 
           col = n)) +        # 빈도에 따라 색깔 표현
 geom_text_wordcloud(seed = 1234) +  
 scale_radius(limits = c(3, NA),
              range = c(3, 30)) +
 scale_color_gradient(low = "#66aaf2",     # 최소 빈도 색깔
                      high = "#004EA1") +  # 최고 빈도 색깔
 theme_minimal()                           # 배경 없는 테마 적용


# 구글 폰트 불러오기
#install.packages("jsonlite")
#install.packages("curl")
#install.packages("showtext")
library(showtext)
font_add_google(name = "Nanum Gothic", family = "nanumgothic")
showtext_auto()

# 워드클라우드 폰트 지정
ggplot(word02_frequency,
       aes(label = word,
           size = n,
           col = n)) +
  geom_text_wordcloud(seed = 1234,
                      family = "nanumgothic") +  # 폰트 적용
  scale_radius(limits = c(3, NA),
               range = c(3, 30)) +
  scale_color_gradient(low = "#66aaf2",
                       high = "#004EA1") +
  theme_minimal()


# 검은 고딕 폰트 지정
font_add_google(name = "Black Han Sans", family = "blackhansans")
showtext_auto()

ggplot(word02_frequency,
       aes(label = word,
           size = n,
           col = n)) +
  geom_text_wordcloud(seed = 1234,
                      family = "blackhansans") +  # 폰트 적용
  scale_radius(limits = c(3, NA),
               range = c(3, 30)) +
  scale_color_gradient(low = "#66aaf2",
                       high = "#004EA1") +
  theme_minimal()

# ggplot2 패키지로 만든 막대 그래프의 폰트 바꾸기 (감자꽃)
font_add_google(name = "Gamja Flower", family = "gamjaflower")
showtext_auto()

ggplot(top20, aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = n), hjust = -0.3) +
  
  labs(title = "문재인 대통령 출마 연설문 단어 빈도",
       x = NULL, y = NULL) +
  
  theme(title = element_text(size = 12),
        text = element_text(family = "gamjaflower"))  # 폰트 적용

## 더 예쁜 워드클라우드 그리기

# 라이브러리 설치
#install.packages('remotes')
library(remotes)
#remotes::install_github("lchiffon/wordcloud2")
library(wordcloud2)

# 라이브러리 설치
wordcloud2(data=word02_frequency, fontFamily = '나눔바른고딕')
wordcloud2(data=word02_frequency, color = "random-light", backgroundColor = "grey", fontFamily = '나눔바른고딕')
