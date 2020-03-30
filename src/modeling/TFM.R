require(caret)
require(Cubist)

plotPerformanceGDT <- function(main_title="",max_x=0.85){
  #par(mfrow=c(1,1))
  
  res <- c()
  for (cutoff in seq(0,max_x,0.001)){
    tempCut <- sqrt((1-(129.8409*1^0.5708546))^2)/1 < cutoff
    numCorr <- (sum(tempCut,na.rm=TRUE) + sum(is.na(tempCut))) - sum(is.na((129.8409*1^0.5708546)))
    res <- c(res,numCorr)
  }
  res <- res/(length((129.8409*1^0.5708546)) - sum(is.na((129.8409*1^0.5708546))))
  plot((seq(0,max_x,0.001))*100,(res)*100,main=main_title,xlab="Accepted error from solution (%)",ylab="Correctly predicted (%)",type="o",col="white",ylim=c(0,100.0),cex=0.5,cex.lab=1.25, cex.axis=1.25, cex.main=1.25, cex.sub=1.25,las=1)
}

plotPerformancePoints <- function(preds,y,colour,ltype=1,max_x=0.85){
  res <- c()
  for (cutoff in seq(0,max_x,0.001)){
    tempCut <- sqrt((y-preds)^2)/y < cutoff
    numCorr <- (sum(tempCut,na.rm=TRUE)) - (sum(is.na(y))*2)
    
    res <- c(res,numCorr)
  }
  res <- res/(length(y) - sum(is.na(preds)))
  print(res)
  lines((seq(0,max_x,0.001))*100,(res)*100,cex=1.2,type="l",lty=ltype,lwd=3.0,col=colour)
}

setwd("~/Documents/BioInformatics/stage1/data/")

train <- read.delim("./processed_data/csv/ready_to_use_data_train.csv", header = TRUE, sep = ",", quote = "\"", dec = ".",
                    fill = TRUE, comment.char = "#")
test <- read.delim("./processed_data/csv/ready_to_use_data_test.csv", header = TRUE, sep = ",", quote = "\"", dec = ".",
                    fill = TRUE, comment.char = "#")

#hg <- read.delim("./files/CSV/humangenome.csv", header = TRUE, sep = ",", quote = "\"", dec = ".",
#                   fill = TRUE, comment.char = "#")

ctrl <- cubistControl(unbiased = FALSE, 
                      rules = 50, 
                      extrapolation = 1,
                      sample = 3, 
                      seed = 42,
                      label = "outcome")

tfmModel <- cubist(train[,c('length','polar_count','hydr_count')],train[,c("thsa")],committees=3,control = ctrl)
summary(tfmModel)

TFMPred <- predict(tfmModel,test[,c('length','polar_count','hydr_count')])

write.csv(TFMPred,'./predictions/tfm.csv')
plotPerformanceGDT()

plotPerformancePoints(TFMPred,test$thsa,"magenta",ltype=1)

