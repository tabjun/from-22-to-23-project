# 5463360 윤태준
setwd('C:\\Users\\Owner\\Desktop')
bruce <- readLines('br.txt',encoding = 'UTF-8')
library(stringr)
library(tidytext)
library(dplyr)
library(lubridate)
head(bruce)
br <-bruce %>% str_replace_all('[??-?R]'," ")
str_squish(br)
head(br)
br <- as_tibble(br)
head(br)
#task1
str_count(br,'gung fu')

#task2
gsub('gung fu','Gung Fu',br)

#task3
e_br <- br %>% unnest_tokens(input = value,
                             output = word,
                             token = 'sentences')
View(e_br)
wo_br<-br %>% unnest_tokens(input = value,
                     output = word,
                     token = 'words')
wo_br
View(wo_br)
ch_br<-br %>% unnest_tokens(input = value,
                            output = word,
                            token = 'characters')

View(ch_br)
which(wo_br == 'kune')
wo_br[1,]
ch_br[10,]
wo_br[117,]
wo_br[10,]
wo_br[68,]
wo_br[69,]
wo_br[70,]
wo_br[163,]
ch_br[24,]
ch_br[15,]
ch_br[19,]
paste(ch_br[24,],
      ch_br[15,],
      ch_br[19,])
# task3 answer
paste(wo_br[1,],' ,',ch_br[10,],wo_br[117,],wo_br[10,],wo_br[68,],wo_br[69,],wo_br[70,],
      ' ,',wo_br[163,],ch_br[24,],ch_br[15,],ch_br[19,],' children',wo_br[137,],wo_br[30,],wo_br[295,])

# task4 
which(grepl('born',e_br))
e_br[1,] 
e_br[8,]
e_br[16,]

#task 4 answer
print('bruce lee was born in 1940year november 27')
print('brandon was born in 1965year february 1')
print('shannon was born in 1969year april 19')
print('die :  july 20, 1973,')

# task 5
birth_br <- paste(wo_br[16,],'11',wo_br[15,])
bb <- ymd(birth_br)
bb
die_br <- paste(wo_br[356,],wo_br[354,],wo_br[355,])
d_br <- ymd(die_br)
d_br
# task 5 answer
interval(bb,d_br) %>% as.period()
