# 03-4, TD-IDF 구하기

# 라이브러리 설치 및 로딩
install.packages("readr")
library(readr)
library(dplyr)
library(stringr)
library(KoNLP)
library(tidytext)
library(ggplot2)

# 데이터 불러오기
raw_speeches <- read_csv("speeches_presidents.csv")
raw_speeches

# 기본적인 전처리
speeches <- raw_speeches %>%
  mutate(value = str_replace_all(value, "[^가-힣]", " "),
         value = str_squish(value))

# 토큰화
speeches <- speeches %>%
  unnest_tokens(input = value,
                output = word,
                token = extractNoun)

# 단어 빈도 구하기
frequecy <- speeches %>%
  count(president, word) %>%
  filter(str_count(word) > 1)

frequecy

# TF-IDF 구하기
frequecy <- frequecy %>%
  bind_tf_idf(term = word,           # 단어
              document = president,  # 텍스트 구분 변수
              n = n) %>%             # 단어 빈도
  arrange(-tf_idf)

frequecy

# TF-IDF가 높은 단어 살펴보기
frequecy %>% filter(president == "문재인")
frequecy %>% filter(president == "박근혜")
frequecy %>% filter(president == "이명박")
frequecy %>% filter(president == "노무현")

# TF-IDF가 낮은 단어 살펴보기
frequecy %>%
  filter(president == "문재인") %>%
  arrange(tf_idf)

frequecy %>%
  filter(president == "박근혜") %>%
  arrange(tf_idf)

# TF-IDF 막대 그래프 작성
# 주요 단어 추출
top10 <- frequecy %>%
  group_by(president) %>%
  slice_max(tf_idf, n = 10, with_ties = F)

# 그래프 순서 정하기
top10$president <- factor(top10$president,
                          levels = c("문재인", "박근혜", "이명박", "노무현"))

# 막대 그래프 작성
ggplot(top10, aes(x = reorder_within(word, tf_idf, president),
                  y = tf_idf,
                  fill = president)) +  
  geom_col(show.legend = F) +
  coord_flip() +
  facet_wrap(~ president, scales = "free", ncol = 2) +
  scale_x_reordered() +
  labs(x = NULL) +
  theme(text = element_text(family = "nanumgothic"))
