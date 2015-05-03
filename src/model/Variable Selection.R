# STAT149
# Shi (Stephen) Fang
setwd('/Users/sfang/Dropbox/Courses/Spring 2015/AM207/Project')
require(gam)
require(rpart)
require(rpart.plot)
require(randomForest)
require(boot)
save = T # save plots?

data = read.csv('Data/games_2015_tournament.csv',header=T,as.is=T)
data = subset(data, game_group==1)
data = within(data, {
  location_Home <- location_Home + location_SemiHome
  location_Away <- location_Away + location_SemiAway
  win <- factor(win)
})

train = subset(data, ncaa_tournament==0)
n_train = nrow(train); n_train

test = subset(data, ncaa_tournament==1)
n_test = nrow(test); n_test

# run random forest to see importance
features = grep('^diff_[a-zA-Z]+$', names(data), value=T) # excl. higher order terms
f = formula(paste0('win~location_Home+location_Away+',
                   paste(features, collapse='+')))
rf0 = randomForest(f,data=train,ntree=500,importance=T)
# plot importance
rf0_importance = sort(importance(rf0,type=1)[,1], decreasing=F)
if (save) {
  filename = "Plots/RF Importance.png"
  png(filename,height=200,width=300,pointsize=9)
}
par(mar=c(4, 9, 2, 2), oma=rep(0, 4), cex=0.8)
barplot(rf0_importance,las=1,horiz=T,
        main='Random Forest Predictor Importance',xlab='Importance')
if (save) dev.off()

# # run decision tree on 2015 regular season
# tree0 = rpart(f, data=train, method="class", parms=list(split="gini"))
# tree0 = rpart(f, data=train, method="class", parms=list(split="gini"),
#               control=rpart.control(minsplit=2,cp=0.001))
# prp(tree0,type=0,extra=106,digits=4)
# if (save) {
#   filename = "Class Tree CP.png"
#   png(filename,height=200,width=300,pointsize=10)
# }
# par(mar=c(4, 4, 2, 2), oma=rep(0, 4), cex=1)
# # plotcp(tree0,ylim=c(0.4,0.8))
# plotcp(tree0)
# if (save) dev.off()
# 
# # prune, print, and display tree
# tree1 = prune(tree0, cp=0.027)
# tree1
# if (save) {
#   filename = "Class Tree Pruned.png"
#   png(filename,height=200,width=300,pointsize=10)
# }
# par(mar=c(4, 4, 2, 2), oma=rep(0, 4), cex=1)
# prp(tree1,type=0,extra=106,digits=4)
# if (save) dev.off()
# 
# # fit gam
# gam0 = gam(win~s(diff_Pythag), data=train, family=binomial)
# summary(gam0)
# 
# if (save) {
#   filename = "GAM Smooths.png"
#   png(filename,height=200,width=300,pointsize=10)
# }
# par(mfrow=c(3,1), mar=c(4, 4, 2, 2), oma=rep(0, 4), cex=1)
# plot(gam0, resid=T, rug=F, se=T, pch=20, col="red")
# if (save) dev.off()
# 
# # test smoothing effect
# glm0 = glm(win~diff_Pythag, data=train, family=binomial)
# anova(glm0, gam0, test='Chi')
# 
# # add 2nd and 3rd order terms
# glm1 = glm(win~diff_Pythag+diff_Pythag.2, data=train, family=binomial)
# glm2 = glm(win~diff_Pythag+diff_Pythag.2+diff_Pythag.3, 
#            data=train, family=binomial)
# glm3 = glm(win~diff_Pythag+diff_Pythag.2+diff_Pythag.3+diff_Pythag.4, 
#            data=train, family=binomial)
# anova(glm0, glm1, test='Chi') # 2nd not significant
# anova(glm0, glm2, test='Chi') # 3rd significant relative to null
# anova(glm2, glm3, test='Chi') # 4th not significant relative to 3rd
# 
# # add up to 3rd order
# pred.x = test[,-which(names(test)=='win')]
# pred.y = test[,'win']
# pred = predict(glm2, pred.x, type='response')
# pred = ifelse(pred > 0.5,1,0)
# accuracy = mean(pred==pred.y)

# cost function
accuracy = function(y,pred) {
  mean(y==ifelse(pred>0.5,1,0))
}

run = T
# use list of most important features for variable selection
if (run) {
  rf0_importance = sort(rf0_importance, decreasing=T)
  scores = rep(NA,length(rf0_importance))
  start = Sys.time()
  for (i in 1:length(rf0_importance)) {
    print(paste0('i=',i))
    features = names(rf0_importance)[1:i]
    f = formula(paste0('win~', paste(features, collapse='+')))
    glm.mod <- glm(f,data=train,family=binomial)
    # k-fold cross validation to select features
    scores[i] = cv.glm(train, glm.mod, cost=accuracy, K=10)$delta[1]
  }
  end = Sys.time()
  print(runtime <- end-start)  
}

# plot scores
if (save) {
  filename = "Plots/XVal Scores.png"
  png(filename,height=200,width=300,pointsize=10)
}
par(mar=c(4, 4, 2, 2), oma=rep(0, 4), cex=1)
plot(scores,type='b',pch=20,
     main='10-Fold Cross-Validation Results',
     ylab='Cross-Validation Score',
     xlab='Number of Most Important Predictors')
if (save) dev.off()

# # test set
# out.scores = rep(NA,length(rf0_importance))
# for (i in 1:length(rf0_importance)) {
#   features = names(rf0_importance)[1:i]
#   f = formula(paste0('win~', paste(features, collapse='+')))
#   glm.mod <- glm(f,data=train,family=binomial)
#   pred.x = test[,-which(names(test)=='win')]
#   pred.y = test[,'win']
#   pred = predict(glm.mod, pred.x, type='response')
#   pred = ifelse(pred > 0.5,1,0)
#   accuracy = mean(pred==pred.y)
#   out.scores[i] = accuracy
# }
# plot(out.scores,type='b',pch=20,ylim=c(0.7,0.8))
# points(scores, type='b',lty=2,pch=1,col='red')

# get priors
i = 5 #0.7611940
features = names(rf0_importance)[1:i]
print(features)
data = read.csv('Data/games.csv',header=T,as.is=T)
data = subset(data, game_group==1)
data = within(data, {
  location_Home <- location_Home + location_SemiHome
  location_Away <- location_Away + location_SemiAway
  win <- factor(win)
})

f = formula(paste0('win~', paste(features, collapse='+')))
glm.prior = glm(f,family=binomial,data=data,subset=year==2014)
priors = summary(glm.prior)$coefficients
write.csv(priors,file='Priors.csv')

# try MCMC




