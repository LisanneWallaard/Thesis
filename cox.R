# code is from: https://www.kaggle.com/code/fangya/machine-learning-survival-for-heart-failure/script
# data is from: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data?resource=download

install.packages('DT')
install.packages('survminer')
install.packages('ggfortify')
install.packages('gtsummary')
library(DT)
library(survminer)
library(tidyverse)
library(survival)
library(ggfortify)
library(gtsummary)
heart<-read.csv("data/heart_failure_clinical_records_dataset.csv")

# Assign ID
heart$id <- seq.int(nrow(heart))

# Assign Character value to Numeric variables
heart$sexc <-ifelse(heart$sex==1, "Male", "Female")
heart$smoke <-ifelse(heart$smoking==1, "Yes", "No")
heart$hbp <- ifelse(heart$high_blood_pressure==1, "Yes","No")
heart$dia <-ifelse(heart$diabetes==1, "Yes", "No")
heart$anaemiac <- ifelse(heart$anaemia==1 ,"Yes", "No")
# Platelets : Hopkins Medicine
heart$platc <- ifelse(heart$platelets>150000 & heart$platelets <450000, "Platelets Normal", "Platelets Abnormal")
heart$plat <- ifelse(heart$platelets>150000 & heart$platelets <450000, 0,1)

# Serum Sodium: Mayo Clinic
heart$sodiumc <- ifelse(heart$serum_sodium >135 & heart$serum_sodium<145, "Serum Sodium Normal", "Serum Sodium Abnormal")
heart$sodiumn <- ifelse(heart$serum_sodium >135 & heart$serum_sodium<145, 0, 1)

#Creatine Phosphkinase : Mountsinai
heart$cpk <- ifelse(heart$creatinine_phosphokinase >10 & heart$creatinine_phosphokinase<120, "CPK Normal", "CPK Abnormal")
heart$cpkn <- ifelse(heart$creatinine_phosphokinase >10 & heart$creatinine_phosphokinase<120, 0, 1)

#ejection_fraction: Mayo
heart$efraction <-ifelse(heart$ejection_fraction<=75 & heart$ejection_fraction>=41, "Ejection Normal", "Ejection Abnormal")
heart$efractionn <-ifelse(heart$ejection_fraction<=75 & heart$ejection_fraction>=41, 0, 1)

#serum_creatinine :mayo
heart$screat<- ifelse((heart$serum_creatinine<1.35 & heart$serum_creatinine>0.74 & heart$sex==1 ) | (heart$serum_creatinine<1.04 & heart$serum_creatinine>0.59 & heart$sex==0) , "Creatinine Normal", "Creatinine Abnormal"   )
heart$screatn<- ifelse((heart$serum_creatinine<1.35 & heart$serum_creatinine>0.74 & heart$sex==1 ) | (heart$serum_creatinine<1.04 & heart$serum_creatinine>0.59 & heart$sex==0) , 0, 1 )

#age group: Pharma convention  
heart$agegp <- ifelse( heart$age<65, "Age <65", "Age >=65")
heart$agegpn <- ifelse( heart$age<65, 0, 1)

#event vs censor
heart$cnsr <- ifelse(heart$DEATH_EVENT==0, "Censor", "Event")

heart

## Original Data table 
h1<- subset(heart, select=c(age,anaemia,creatinine_phosphokinase, serum_creatinine,diabetes, ejection_fraction ,high_blood_pressure, platelets , serum_sodium, sex, smoking, DEATH_EVENT))
head(h1, 5) %>% DT::datatable()

## Modified Categorical Data table
h1c<- subset(heart, select=c(agegp,anaemiac,cpk, screat, dia, efraction ,hbp, platc, sodiumc, sexc, smoke, DEATH_EVENT, time))
head(h1c, 5)%>% DT::datatable()

#Modified Categorical variable selection
m1<- subset(heart, select=c(agegpn,anaemia,cpkn, screatn, diabetes, efractionn ,high_blood_pressure, plat, sodiumn, sex, smoking, DEATH_EVENT))


## Training + Testing Data
set.seed=8
train.test.split<-sample(2, nrow(h1), replace=TRUE, prob=c(0.8,0.2))
train=h1[train.test.split==1,]
test=h1[train.test.split==2,]

set.seed=18
train.test.split1<-sample(2, nrow(m1), replace=TRUE, prob=c(0.7,0.3))
train1=m1[train.test.split==1,]
test1=m1[train.test.split==2,]

head(train, 5)%>% DT::datatable()
head(test, 5)%>% DT::datatable()

# Survival Analysis

# From applying various Machine Learning models, we reached a similar result, 
# that Serum Sodium, Serum Creatinine, Ejection Fraction, and Age Group are the most important features in Heart Disease.
cox_cat <- coxph(Surv(time,DEATH_EVENT) ~ efraction + agegp + screat + sodiumc, data=heart)
cox_bin <- coxph(Surv(time,DEATH_EVENT) ~ efractionn + agegpn + screatn + sodiumn, data=heart)
cox_num <- coxph(Surv(time,DEATH_EVENT) ~ ejection_fraction + age + serum_creatinine + serum_sodium, data=heart)

X_cat <- data.frame(time = 3650, DEATH_EVENT = 0, efraction = "Ejection Abnormal", agegp = "Age >=65", screat = "Creatinine Abnormal", sodiumc = "Serum Sodium Abnormal")
X_bin <- data.frame(time = 3650, DEATH_EVENT = 0, efractionn = 1, agegpn = 1, screatn = 1, sodiumn = 1)
X_num <- data.frame(time = 3650, DEATH_EVENT = 0, ejection_fraction = 80, age = 80, serum_creatinine = 9, serum_sodium = 150)

surv_prob_cat <- predict(cox_cat, newdata = X_cat, type = "survival")
surv_prob_bin <- predict(cox_bin, newdata = X_bin, type = "survival")
surv_prob_num <- predict(cox_num, newdata = X_num, type = "expected")

#Save the model as a file
saveRDS(cox_num, file = "model/cox_num.rds")