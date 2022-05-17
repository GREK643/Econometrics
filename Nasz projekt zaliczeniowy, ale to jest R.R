bazadanych<-read.csv(file="C:/Users/Kacper/Documents/Ekonometria Laby/projekt/income2wampoprzeksztalceniach.csv",
                     header=T,sep=";") 
bazadanych$X<-NULL
bazadanych$wage_per_hour<-NULL
View(bazadanych)
lm(lwage~.,data=bazadanych)
regression<-lm(lwage~.,data=bazadanych)
summary(regression)
View(regression)

###################################### licencjat###################################### 


library(readr)
eurostat<-read_tsv(file="C:/Users/Kacper/OneDrive/Dokumenty/Licencjat/nasa_10_f_bs.tsv/nasa_10_f_bs.tsv",
                    col_names=T,) 
colnames
eurostat$`unit,co_nco,sector,finpos,na_item,geo\time`<-NULL
regression<-lm(2020 ~.,data=eurostat)
