library(tidyverse)
f <-structure(list(.id = c("EOB14", "EOB14", "EOB15", "EOB15", "EOCAR", "EOCAR", "EOCAL", 
                           "EOCAL", "EOC2R", "EOC2R", "EOC2L", "EOC2L", "DCCT_DN", "DCCT_DN"), 
                   x = c(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1), 
                   Freq = c(8951L, 1205L, 10113L, 53L, 3104L, 7012L, 3102L, 7008L, 10050L, 66L, 10029L, 61L, 8433L, 1943L)), 
                  row.names = c(31L, 32L, 33L, 34L, 35L, 36L, 37L, 38L, 49L, 50L, 51L, 52L, 102L,  103L), 
                  class = "data.frame")

ff <-  f %>% filter(!((x == 2 & Freq < 200) | (x == 1 & Freq > 10000)))
ff


library(tidyverse)
df <- data.frame(mynumber = c(1,2,3), trend = c(4,3,1))
new_df <-  df %>% mutate(mynumber = if_else(trend ==4 | trend == 1, 'google', as.character(mynumber)))
new_df