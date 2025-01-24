
rm(list = ls())

setwd("~/Dropbox/Research/AbbVie/Interpretation/3_final_v3//4_Sim_3/")

library(tensorflow)
library(keras)
library(doParallel)
library(optimg)
library(xtable)
library(BOIN)

###################################################################
f1.l2.func = function(w.in, a.in, b.in){
  f1.line = c(0, a.in*w.in[1]+b.in*w.in[1])
  return(exp(f1.line)/(sum(exp(f1.line))))
}

f2.l2.func = function(w.in, a.in, b.in){
  f1.line = c(0, a.in*w.in[1]+b.in*w.in[2])
  return(exp(f1.line)/(sum(exp(f1.line))))
}

f3.l2.func = function(w.in, a.in, b.in){
  f1.line = c(0, a.in*w.in[1]+b.in*w.in[2]+w.in[3])
  return(exp(f1.line)/(sum(exp(f1.line))))
}

f1.l1.func = function(w.in, a.in, b.in, c.in){
  f1.return.vec = c(
    (w.in[1]*a.in+w.in[1]*b.in)/c.in
  )
  
  return(f1.return.vec)
  
}

f2.l1.func = function(w.in, a.in, b.in, c.in){
  f1.return.vec = c(
    (w.in[1]*a.in+w.in[2]*b.in)/c.in
  )
  
  return(f1.return.vec)
}

f3.l1.func = function(w.in, a.in, b.in, c.in, d.in, e.in, f.in){
  f1.return.vec = c(
    (w.in[1]*a.in+w.in[2]*b.in)/(c.in+w.in[3])
  )
  
  return(f1.return.vec)
}


# f1.l2.func = function(w.in, a.in, b.in){
#   f1.line = w.in[1]*(a.in+b.in)
#   return(exp(f1.line)/(sum(exp(f1.line))))
# }
# 
# f2.l2.func = function(w.in, a.in, b.in){
#   f1.line = w.in[1]*a.in+w.in[2]*b.in
#   return(exp(f1.line)/(sum(exp(f1.line))))
# }
# 
# f3.l2.func = function(w.in, a.in, b.in){
#   f1.line = w.in[1]*a.in+w.in[2]*b.in+w.in[3]
#   return(exp(f1.line)/(sum(exp(f1.line))))
# }
# 
# f1.l1.func = function(w.in, a.in, b.in, c.in){
#   f1.return.vec = c(
#     w.in[1]*a.in/c.in,
#     w.in[1]*b.in/c.in
#   )
#   
#   return(f1.return.vec)
# 
# }
# 
# f2.l1.func = function(w.in, a.in, b.in, c.in){
#   f1.return.vec = c(
#     w.in[1]*a.in/c.in,
#     w.in[2]*b.in/c.in
#   )
#   
#   return(f1.return.vec)
# }
# 
# f3.l1.func = function(w.in, a.in, b.in, c.in, d.in, e.in, f.in){
#   f1.return.vec = c(
#     w.in[1]*a.in/c.in+w.in[3],
#     w.in[2]*b.in/c.in
#   )
#   
#   return(f1.return.vec)
# }


###################################################################
## parameters
set.seed(1)
n.itt = 10^3
n.cluster = 8
MTD.target = 0.3
n.dose = 2
n.max = 30

x.train.mat = matrix(NA, nrow = n.itt, ncol = n.dose+1)
dec.vec = rep(NA, n.itt)
include.ind = TRUE

for (itt in 1:n.itt) {

  set.seed(itt)
  
  n.group = round(runif(1, 10, n.max))
  x.pbo = round(runif(1, 1, n.group))
  x.trt = round(runif(1, 1, n.group))
  
  x.train.mat[itt, ] = c(x.pbo, x.trt, n.group)
  
  fisher.data = matrix(c(x.pbo, x.trt, n.group - x.pbo, n.group - x.trt),
                       nrow = 2, ncol = 2, byrow=TRUE)
  fisher.p.value.train = fisher.test(fisher.data,
                                     alternative = "less")$p.value
  
  dec.vec[itt] = fisher.p.value.train<0.05
  
}
print(table(dec.vec))

#####################################################################
## DNN training
data.input.all = cbind(x.train.mat)

data.label.all = matrix(0, nrow = n.itt, ncol = n.dose)
data.label.all[col(data.label.all) == (dec.vec+1)] = 1

#############################################################
## 5-fold cross-validation
n.itt = dim(data.label.all)[1]
n.cross = 5
cross.tab.out = matrix(NA, nrow = n.cross, ncol = 4)
for (cross.ind in 1:n.cross){
  val.index = 1:(n.itt/n.cross) + (cross.ind -1)*n.itt/n.cross
  train.index = (1:n.itt)[-val.index]
  
  data.input.train = data.input.all[train.index, ]
  data.label.train = data.label.all[train.index, ]
  
  data.input.val = data.input.all[val.index, ]
  data.label.val = data.label.all[val.index, ]
  
  set_random_seed(1)
  model <- keras_model_sequential()
  
  model %>%
    layer_dense(units = 60, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 60, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 2, activation = 'softmax')
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.0005),
    loss = 'categorical_crossentropy',
    metrics = list('accuracy')
  )
  
  dnn_history = model %>% fit(
    data.input.train,
    data.label.train,
    epochs = 100,
    batch_size = 10,
    validation_split = 0
  )
  
  # model prediction
  train.pred = (model %>% predict(data.input.train))
  # train.pred[train.pred<10^(-6)]=10^(-6)
  cross.tab.out[cross.ind, 1] = 
    mean(-apply(data.label.train*log(train.pred), 1, sum))
  
  cross.tab.out[cross.ind, 3] = mean(
    (apply(train.pred, 1, function(x){which.max(x)}))==
      (apply(data.label.train, 1, function(x){which.max(x)}))
  )
  
  val.pred = (model %>% predict(data.input.val))
  cross.tab.out[cross.ind, 2] = 
    mean(-apply(data.label.val*log(val.pred), 1, sum))
  
  cross.tab.out[cross.ind, 4] = mean(
    (apply(val.pred, 1, function(x){which.max(x)}))==
      (apply(data.label.val, 1, function(x){which.max(x)}))
  )
  
}

DNN.train.MSE = mean(cross.tab.out[,1])
DNN.val.MSE = mean(cross.tab.out[,2])
DNN.train.acc = mean(cross.tab.out[,3])
DNN.val.acc = mean(cross.tab.out[,4])

#############################################################################
## simple models
f.func.value = function(l2.ind.in, l1.1.ind.in, l1.2.ind.in, 
                        w.in,
                        x.total.in){
  if (l2.ind.in==1){
    l2.f.ind = f1.l2.func
    l2.w.in = w.in[1:1]
  } else if (l2.ind.in==2){
    l2.f.ind = f2.l2.func
    l2.w.in = w.in[1:2]
  } else if (l2.ind.in==3){
    l2.f.ind = f3.l2.func
    l2.w.in = w.in[1:3]
  } 
  
  if (l1.1.ind.in==1){
    l1.1.f.ind = f1.l1.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+1)]
  } else if (l1.1.ind.in==2){
    l1.1.f.ind = f2.l1.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+2)]
  } else if (l1.1.ind.in==3){
    l1.1.f.ind = f3.l1.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+3)]
  } 
  
  if (l1.2.ind.in==1){
    l1.2.f.ind = f1.l1.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+1)]
  } else if (l1.2.ind.in==2){
    l1.2.f.ind = f2.l1.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+2)]
  } else if (l1.2.ind.in==3){
    l1.2.f.ind = f3.l1.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+3)]
  } 
  
  f.total.temp = l2.f.ind(l2.w.in, 
                          l1.1.f.ind(l1.1.w.in, x.total.in[1], x.total.in[2],
                                     x.total.in[3]),
                          l1.2.f.ind(l1.2.w.in, x.total.in[1], x.total.in[2],
                                     x.total.in[3]))
  return(f.total.temp)
  
}

f.func.mse = function(l2.ind.mse.in, l1.1.ind.mse.in, l1.2.ind.mse.in, 
                      w.mse.in, 
                      x.total.mse.in, y.total.mse.in){
  
  # f.total.temp.mse = f.func.value(l2.ind.mse.in, l1.1.ind.mse.in, 
  #                                 l1.2.ind.mse.in, 
  #                                 w.mse.in,
  #                                 x.total.mse.in)
  
  f.total.temp.mse = t(sapply(1:(dim(x.total.mse.in)[1]),
                            function(f.temp.ind){f.func.value(
                              l2.ind.mse.in, l1.1.ind.mse.in, 
                              l1.2.ind.mse.in,
                              w.mse.in,
                              x.total.mse.in[f.temp.ind,])}))
  
  mse.out = mean(-apply(y.total.mse.in*
                log(f.total.temp.mse), 1, sum))
  
  return(mse.out)
  
}

f.func.com = function(l2.ind.com.in, l1.1.ind.com.in, l1.2.ind.com.in, 
                      x.train.com.in, y.train.com.in,
                      x.val.com.in, y.val.com.in){
  l2.w.leng = l2.ind.com.in 
  l1.1.w.leng = l1.1.ind.com.in
  l1.2.w.leng = l1.2.ind.com.in
  
  set.seed(1)
  opt.par.init = runif(l2.w.leng+l1.1.w.leng+l1.2.w.leng, 0.2, 0.5)
  # print(opt.par.init)
  opt.par = optimg(opt.par.init, f.func.mse,
                   l2.ind.mse.in = l2.ind.com.in, 
                   l1.1.ind.mse.in = l1.1.ind.com.in,
                   l1.2.ind.mse.in = l1.2.ind.com.in,
                   x.total.mse.in = x.train.com.in, 
                   y.total.mse.in = y.train.com.in,
                   method="ADAM")
  
  
  # pred.train = f.func.value(l2.ind.com.in, l1.1.ind.com.in, l1.2.ind.com.in, 
  #                           opt.par$par, x.train.com.in)
  
  pred.train = t(sapply(1:(dim(x.train.com.in)[1]),
                              function(f.temp.ind){f.func.value(
                                l2.ind.com.in, l1.1.ind.com.in, 
                                l1.2.ind.com.in,
                                opt.par$par,
                                x.train.com.in[f.temp.ind,])}))
  
  pred.train[pred.train<10^(-6)] = 10^(-6) 
  pred.train.loss = mean(-apply(y.train.com.in*
                          log(pred.train), 1, sum))
  
  train.acc = mean(
    (apply(pred.train, 1, function(x){which.max(x)}))==
      (apply(y.train.com.in, 1, function(x){which.max(x)}))
  )
  
  # pred.val = f.func.value(l2.ind.com.in, l1.1.ind.com.in, l1.2.ind.com.in, 
  #                         opt.par$par, x.val.com.in)
  
  pred.val = t(sapply(1:(dim(x.val.com.in)[1]),
                        function(f.temp.ind){f.func.value(
                          l2.ind.com.in, l1.1.ind.com.in, 
                          l1.2.ind.com.in,
                          opt.par$par,
                          x.val.com.in[f.temp.ind,])}))
  
  pred.val[pred.val<10^(-6)] = 10^(-6) 
  
  pred.val.loss = mean(-apply(y.val.com.in*
                    log(pred.val), 1, sum))
  
  val.acc = mean(
    (apply(pred.val, 1, function(x){which.max(x)}))==
      (apply(y.val.com.in, 1, function(x){which.max(x)}))
  )
  
  return(c(pred.train.loss, pred.val.loss, train.acc, val.acc))
  
}

n.cross = 5
test.tab.out = array(NA, dim = c(10, 5, n.cross))
for (cross.ind in 1:n.cross){
  val.index = 1:(n.itt/n.cross) + (cross.ind -1)*n.itt/n.cross
  train.index = (1:n.itt)[-val.index]
  
  data.input.train = data.input.all[train.index, ]
  data.label.train = data.label.all[train.index, ]
  
  data.input.val = data.input.all[val.index, ]
  data.label.val = data.label.all[val.index, ] 
  
  ## simple DNN as a benchmark
  set_random_seed(1)
  simple.model <- keras_model_sequential()
  
  simple.model %>%
    layer_dense(units = 2, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 2, activation = 'softmax')
   
  simple.model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.0005),
    loss = 'categorical_crossentropy',
    metrics = list('accuracy')
  ) 
  
  dnn_simple_history = simple.model %>% fit(
    data.input.train,
    data.label.train,
    epochs = 100,
    batch_size = 10,
    validation_split = 0
  )
  
  # model prediction
  train.pred = (simple.model %>% predict(data.input.train))
  val.pred = (simple.model %>% predict(data.input.val))
  
  DNN.simple.acc.train = mean(
    (apply(train.pred, 1, function(x){which.max(x)}))==
      (apply(data.label.train, 1, function(x){which.max(x)}))
  )
  
  DNN.simple.acc.val = mean(
    (apply(val.pred, 1, function(x){which.max(x)}))==
      (apply(data.label.val, 1, function(x){which.max(x)}))
  )

  test.out.temp = rbind(
  c(f.func.com(1, 1, 2, data.input.train, data.label.train,
             data.input.val, data.label.val), 4/2),
  c(f.func.com(1, 1, 3, data.input.train, data.label.train,
             data.input.val, data.label.val), 5/2),
  c(f.func.com(1, 2, 3, data.input.train, data.label.train,
             data.input.val, data.label.val), 6/2),
  
  c(f.func.com(2, 1, 2, data.input.train, data.label.train,
               data.input.val, data.label.val), 5/2),
  c(f.func.com(2, 1, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 6/2),
  c(f.func.com(2, 2, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 7/2),
  
  c(f.func.com(3, 1, 2, data.input.train, data.label.train,
               data.input.val, data.label.val), 6/2),
  c(f.func.com(3, 1, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 7/2),
  c(f.func.com(3, 2, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 8/2),
  
  c(mean(-apply(data.label.train*log(train.pred), 1, sum)),
    mean(-apply(data.label.val*log(val.pred), 1, sum)),
    DNN.simple.acc.train,
    DNN.simple.acc.val,
    count_params(simple.model)/2
    )
  
  )
  
  test.tab.out[,, cross.ind] = test.out.temp

}

test.out = apply(test.tab.out, 1:2, mean)
print(test.out)

#########################################################################
## final the optimal lambda
lambda.vec = seq(0, 10, length.out = 1000)
lambda.out.vec = rep(NA, length(lambda.vec))

for (lambda.itt in 1:length(lambda.vec)){
  lambda.eval = lambda.vec[lambda.itt]
  
  cor.1.vec = test.out[1:9,1]/DNN.train.MSE - 1 + 
    lambda.eval*test.out[1:9, 5]
  cor.2.vec = test.out[1:9,2]/DNN.val.MSE - 1 + 
    lambda.eval*test.out[1:9, 5]
  
  lambda.out.vec[lambda.itt] = cor(cor.1.vec, cor.2.vec, method = "spearman")
  
}

print(unique(lambda.out.vec))
# print(lambda.out.vec)

lambda.opt = lambda.vec[which.max(lambda.out.vec)]
print(lambda.opt)

print(which.min(test.out[,1]/DNN.train.MSE - 1 + lambda.opt*test.out[,5]))
print(which.min(test.out[,2]/DNN.val.MSE - 1 + lambda.opt*test.out[,5]))
print(which.min(test.out[,1]))
print(which.min(test.out[,2]))

print(cor(test.out[,1], test.out[,2], method = "spearman"))
print(cor(test.out[,1]/DNN.train.MSE - 1 + lambda.opt*test.out[,5],
          test.out[,2]/DNN.val.MSE - 1 + lambda.opt*test.out[,5],
          method = "spearman"))

write.csv(lambda.opt, paste0("lambda_opt.csv"))


###############################################
## final results
final.results.out = data.frame(test.out)
final.results.out = rbind(final.results.out,
                          c(round(DNN.train.MSE, 4),
                            round(DNN.val.MSE, 4), 
                            round(DNN.train.acc, 4), 
                            round(DNN.val.acc, 4), 
                            count_params(model)/3))
colnames(final.results.out) = c("MSE.train", "MSE.val", 
                                "acc.train", "acc.val", "p")
final.results.out$cp.train = final.results.out$MSE.train/DNN.train.MSE - 1 + 
  lambda.opt*final.results.out$p
final.results.out$cp.val = final.results.out$MSE.val/DNN.val.MSE - 1 + 
  lambda.opt*final.results.out$p

final.results.out = final.results.out[,c(5, 1, 2, 6, 7, 3, 4)]

write.csv(final.results.out, paste0("sim_3.csv"))

## latex table 

# final.results.out = round(final.results.out, 5)
# final.results.out = rbind(final.results.out, 
#                           c(count_params(model), round(DNN.train.MSE, 5), 
#                             round(DNN.val.MSE, 5), "", ""))

print(xtable(final.results.out, digits = c(0, 0, 3, 3, 1, 1, 3, 3)), 
      include.rownames = TRUE)











