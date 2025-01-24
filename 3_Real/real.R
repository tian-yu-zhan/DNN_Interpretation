
setwd("~/Dropbox/Research/AbbVie/Interpretation/3_final_v3/3_Real/")
library(haven)
library(tidyr)
library(tensorflow)
library(keras)
library(doParallel)
library(optimg)
library(xtable)

###################################################################

f1.func = function(w.in, a.in, b.in){
  return(w.in[1]*(a.in + b.in)^2 )
}

f2.func = function(w.in, a.in, b.in){
  return(w.in[1]*(a.in^3)+w.in[2]*(b.in^3))
}

f3.func = function(w.in, a.in, b.in){
  return(w.in[1]*(a.in)+w.in[2]*(b.in)+w.in[3])
}

#################################################################
data.1 = read_xpt("data/ALB_CR_J.XPT")
data.1.com = data.1[, c("SEQN", "URXUMA", "URXUCR")]

data.2 = read_xpt("data/HDL_J.XPT")
data.2.com = data.2[, c("SEQN", "LBDHDD")]

data.3 = read_xpt("data/HSCRP_J.XPT")
data.3.com = data.3[, c("SEQN", "LBXHSCRP")]

data.4 = read_xpt("data/INS_J.XPT")
data.4.com = data.4[, c("SEQN", "LBXIN")]

data.5 = read_xpt("data/TRIGLY_J.XPT")
data.5.com = data.5[, c("SEQN", "LBXTR", "LBDLDL")]

data.6 = read_xpt("data/UIO_J.XPT")
data.6.com = data.6[, c("SEQN", "URXUIO")]

data.7 = read_xpt("data/VIC_J.XPT")
data.7.com = data.7[, c("SEQN", "LBXVIC")]

data.8 = read_xpt("data/VID_J.XPT")
data.8.com = data.8[, c("SEQN", "LBXVD2MS", "LBXVD3MS")]

## merge datasets
df_list <- list(data.1.com, data.2.com, data.3.com, 
                data.4.com, data.5.com, data.6.com, 
                data.7.com, data.8.com)      

#merge all data frames together
data.com = Reduce(function(x, y) merge(x, y, by="SEQN"), df_list)  
data.comp = na.omit( data.com)[,-1]

# data.comp = data.comp[data.comp$URXUMA<=200, ]

data.input.all = scale(data.comp[,-1])

data.label.temp = log(data.comp[,1])
data.label.all = (data.label.temp+3) / 14


########################################################################
## DNN training
n.all.itt = round(dim(data.input.all)[1])

## 4-fold cross-validation
n.cross = 4
cross.tab.out = matrix(NA, nrow = n.cross, ncol = 2)
for (cross.ind in 1:n.cross){
  val.index = 1:(n.all.itt/n.cross) + (cross.ind -1)*n.all.itt/n.cross
  train.index = (1:n.all.itt)[-val.index]
  
  data.input.train = data.input.all[train.index, ]
  data.label.train = data.label.all[train.index]
  
  data.input.val = data.input.all[val.index, ]
  data.label.val = data.label.all[val.index]

set_random_seed(1)
model <- keras_model_sequential()

model %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  # layer_dense(units = 40, activation = "relu") %>%
  # layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.005),
  loss = 'mse',
  metrics = list('mse')
)

dnn_history = model %>% fit(
  data.input.train,
  data.label.train,
  epochs = 100,
  batch_size = 10,
  validation_split = 0
)

# model prediction
train.pred <- as.numeric(model %>% predict(data.input.train))
cross.tab.out[cross.ind, 1] = (mean((train.pred-data.label.train)^2))

val.pred <- as.numeric(model %>% predict(data.input.val))
cross.tab.out[cross.ind, 2] = (mean((val.pred-data.label.val)^2))

}

DNN.train.MSE = mean(cross.tab.out[,1])
DNN.val.MSE = mean(cross.tab.out[,2])

#data.input.train.all = data.input.train
#data.input.val.all = data.input.val

print(DNN.train.MSE)
print(DNN.val.MSE)

#############################################################################
## simple models
f.func.value = function(l2.ind.in, l1.1.ind.in, l1.2.ind.in, 
                        w.in,
                        x.total.in){
  if (l2.ind.in==1){
    l2.f.ind = f1.func
    l2.w.in = w.in[1:1]
  } else if (l2.ind.in==2){
    l2.f.ind = f2.func
    l2.w.in = w.in[1:2]
  } else if (l2.ind.in==3){
    l2.f.ind = f3.func
    l2.w.in = w.in[1:3]
  } else if (l2.ind.in==4){
    l2.f.ind = f4.func
    l2.w.in = w.in[1:4]
  }
  
  if (l1.1.ind.in==1){
    l1.1.f.ind = f1.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+1)]
  } else if (l1.1.ind.in==2){
    l1.1.f.ind = f2.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+2)]
  } else if (l1.1.ind.in==3){
    l1.1.f.ind = f3.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+3)]
  } else if (l1.1.ind.in==4){
    l1.1.f.ind = f4.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+4)]
  }
  
  if (l1.2.ind.in==1){
    l1.2.f.ind = f1.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+1)]
  } else if (l1.2.ind.in==2){
    l1.2.f.ind = f2.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+2)]
  } else if (l1.2.ind.in==3){
    l1.2.f.ind = f3.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+3)]
  } else if (l1.2.ind.in==4){
    l1.2.f.ind = f4.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+4)]
  }
  
  f.total.temp = l2.f.ind(l2.w.in, 
                          l1.1.f.ind(l1.1.w.in, x.total.in[,1], x.total.in[,2]),
                          l1.2.f.ind(l1.2.w.in, x.total.in[,1], x.total.in[,2]))
  return(f.total.temp)
  
}

# a = f.func.value(3, 1, 2, opt.par$par,
#                  data.input.val)

f.func.mse = function(l2.ind.mse.in, l1.1.ind.mse.in, l1.2.ind.mse.in, 
                      w.mse.in, 
                      x.total.mse.in, y.total.mse.in){
  
  f.total.temp.mse = f.func.value(l2.ind.mse.in, l1.1.ind.mse.in, 
                                  l1.2.ind.mse.in, 
                                  w.mse.in,
                                  x.total.mse.in)
  mse.out = mean((y.total.mse.in-f.total.temp.mse)^2)
  return(mse.out)
  
}

f.func.com = function(l2.ind.com.in, l1.1.ind.com.in, l1.2.ind.com.in, 
                      x.train.com.in, y.train.com.in,
                      x.val.com.in, y.val.com.in){
  l2.w.leng = l2.ind.com.in 
  l1.1.w.leng = l1.1.ind.com.in
  l1.2.w.leng = l1.2.ind.com.in
  
  # if (l2.ind.com.in==3){
  #   l2.w.leng = 2
  # } else {
  #   l2.w.leng = 3
  # }
  # 
  # if (l1.1.ind.com.in==3){
  #   l1.1.w.leng = 2
  # } else {
  #   l1.1.w.leng = 3
  # }
  # 
  # if (l1.2.ind.com.in==3){
  #   l1.2.w.leng = 2
  # } else {
  #   l1.2.w.leng = 3
  # }
  
  
  set.seed(1)
  opt.par.init = rnorm(l2.w.leng+l1.1.w.leng+l1.2.w.leng, 0, 1)
  opt.par = optimg(opt.par.init, f.func.mse,
                   l2.ind.mse.in = l2.ind.com.in, 
                   l1.1.ind.mse.in = l1.1.ind.com.in,
                   l1.2.ind.mse.in = l1.2.ind.com.in,
                   x.total.mse.in = x.train.com.in, 
                   y.total.mse.in = y.train.com.in,
                   method="ADAM")
  
  pred.train = f.func.value(l2.ind.com.in, l1.1.ind.com.in, l1.2.ind.com.in, 
                            opt.par$par,x.train.com.in)
  pred.val = f.func.value(l2.ind.com.in, l1.1.ind.com.in, l1.2.ind.com.in, 
                          opt.par$par,x.val.com.in)
  
  return(c(mean((pred.train-y.train.com.in)^2),
           mean((pred.val-y.val.com.in)^2)))
  
}

n.cov = dim(data.input.all)[2]
combn.mat = combn(n.cov, 2)

n.cross = 4
test.tab.out = array(NA, dim = c(dim(combn.mat)[2], 9, 3, n.cross))
for (cross.ind in 1:n.cross){
  val.index = 1:(n.all.itt/n.cross) + (cross.ind -1)*n.all.itt/n.cross
  train.index = (1:n.all.itt)[-val.index]
  
  data.input.train.all = data.input.all[train.index, ]
  data.label.train = data.label.all[train.index]
  
  data.input.val.all = data.input.all[val.index, ]
  data.label.val = data.label.all[val.index]

mcp.mat.temp = array(NA, dim = c(dim(combn.mat)[2], 9, 3))

for (cov.itt in 1:(dim(combn.mat)[2])){

print(cov.itt)
data.input.train = data.input.train.all[, combn.mat[, cov.itt]]
data.input.val = data.input.val.all[, combn.mat[, cov.itt]]

mcp.mat.temp[cov.itt, , ] = rbind(
  c(f.func.com(1, 1, 2, data.input.train, data.label.train,
               data.input.val, data.label.val), 4),
  c(f.func.com(1, 1, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 5),
  c(f.func.com(1, 2, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 6),
  
  c(f.func.com(2, 1, 2, data.input.train, data.label.train,
               data.input.val, data.label.val), 5),
  c(f.func.com(2, 1, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 6),
  c(f.func.com(2, 2, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 7),
  
  c(f.func.com(3, 1, 2, data.input.train, data.label.train,
               data.input.val, data.label.val), 6),
  c(f.func.com(3, 1, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 7),
  c(f.func.com(3, 2, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 8)
  
)

}

test.tab.out[,,, cross.ind] = mcp.mat.temp

}

mcp.mat = apply(test.tab.out, 1:3, mean)

#########################################################################
## final the optimal lambda
lambda.vec = seq(0, 10, length.out = 1000)
lambda.out.vec = rep(NA, length(lambda.vec))

MSE.train.vec = as.vector(mcp.mat[,,1])
MSE.val.vec = as.vector(mcp.mat[,,2])
p.vec = as.vector(mcp.mat[,,3])

for (lambda.itt in 1:length(lambda.vec)){
  lambda.eval = lambda.vec[lambda.itt]
  
  cor.1.vec = MSE.train.vec/DNN.train.MSE - 1 + lambda.eval*p.vec
  cor.2.vec = MSE.val.vec/DNN.val.MSE - 1 + lambda.eval*p.vec
  
  lambda.out.vec[lambda.itt] = cor(cor.1.vec, cor.2.vec, method = "spearman")
  
}

print(unique(lambda.out.vec))
# print(lambda.out.vec)

lambda.opt = lambda.vec[which.max(lambda.out.vec)]
print(lambda.opt)

######## selected model
select.method.id = which.min(MSE.val.vec/DNN.val.MSE - 1 + lambda.opt*p.vec)
print(ceiling(select.method.id/(dim(combn.mat)[2])))
print(((select.method.id)%%(dim(combn.mat)[2])))

print((MSE.train.vec/DNN.train.MSE - 1 + lambda.opt*p.vec)[select.method.id])
print((MSE.val.vec/DNN.val.MSE - 1 + lambda.opt*p.vec)[select.method.id])
print((MSE.train.vec)[select.method.id])
print((MSE.val.vec)[select.method.id])

### simple DNN
n.cross = 4
simple.cross.tab.out = matrix(NA, nrow = n.cross, ncol = 2)
for (cross.ind in 1:n.cross){
  val.index = 1:(n.all.itt/n.cross) + (cross.ind -1)*n.all.itt/n.cross
  train.index = (1:n.all.itt)[-val.index]
  
  data.input.train = data.input.all[train.index, ]
  data.label.train = data.label.all[train.index]
  
  data.input.val = data.input.all[val.index, ]
  data.label.val = data.label.all[val.index]
  
  set_random_seed(1)
  simple.model <- keras_model_sequential()
  
  simple.model %>%
    layer_dense(units = 2, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 1, activation = 'sigmoid')
  
  simple.model %>% compile(
    optimizer = optimizer_rmsprop(learning_rate = 0.005),
    loss = 'mse',
    metrics = list('mse')
  )
  
  dnn_simple_history = simple.model %>% fit(
    data.input.train,
    data.label.train,
    epochs = 100,
    batch_size = 10,
    validation_split = 0
  )
  
  # model prediction
  train.pred <- as.numeric(simple.model %>% predict(data.input.train))
  simple.cross.tab.out[cross.ind, 1] = (mean((train.pred-data.label.train)^2))
  
  val.pred <- as.numeric(simple.model %>% predict(data.input.val))
  simple.cross.tab.out[cross.ind, 2] = (mean((val.pred-data.label.val)^2))
  
}

DNN.simple.train.MSE = mean(simple.cross.tab.out[,1])
DNN.simple.val.MSE = mean(simple.cross.tab.out[,2])

print(DNN.simple.train.MSE)
print(DNN.simple.val.MSE)
print((DNN.simple.train.MSE/DNN.train.MSE - 1 + 
         lambda.opt*count_params(simple.model)))
print((DNN.simple.val.MSE/DNN.val.MSE - 1 + 
         lambda.opt*count_params(simple.model)))

### full DNN
print(DNN.train.MSE)
print(DNN.val.MSE)
print((DNN.train.MSE/DNN.train.MSE - 1 + 
         lambda.opt*count_params(model)))
print((DNN.val.MSE/DNN.val.MSE - 1 + 
         lambda.opt*count_params(model)))

## latex table
latex.tab.out = matrix(NA, nrow = 3, ncol = 5)
latex.tab.out[1, ] = c(5,
                       (MSE.train.vec)[select.method.id],
                       (MSE.val.vec)[select.method.id],
                       (MSE.train.vec/DNN.train.MSE - 1 + 
                          lambda.opt*p.vec)[select.method.id],
                       (MSE.val.vec/DNN.val.MSE - 1 + 
                          lambda.opt*p.vec)[select.method.id])

latex.tab.out[2, ] = c(count_params(simple.model),
                       DNN.simple.train.MSE,
                       DNN.simple.val.MSE,
                       (DNN.simple.train.MSE/DNN.train.MSE - 1 + 
                          lambda.opt*count_params(simple.model)),
                       (DNN.simple.val.MSE/DNN.val.MSE - 1 + 
                          lambda.opt*count_params(simple.model)))

latex.tab.out[3, ] = c(count_params(model),
                       DNN.train.MSE,
                       DNN.val.MSE,
                       (DNN.train.MSE/DNN.train.MSE - 1 + 
                          lambda.opt*count_params(model)),
                       (DNN.val.MSE/DNN.val.MSE - 1 + 
                          lambda.opt*count_params(model)))

print(xtable(latex.tab.out, digits = c(0, 0, 4, 4, 1, 1)), 
      include.rownames = TRUE)

### selected model with all data
data.input.final = data.input.all[, 
            combn.mat[, (select.method.id)%%(dim(combn.mat)[2])]]

set.seed(1)
opt.par.init = rnorm(5, 0, 1)
opt.par = optimg(opt.par.init, f.func.mse,
                 l2.ind.mse.in = 1, 
                 l1.1.ind.mse.in = 1,
                 l1.2.ind.mse.in = 3,
                 x.total.mse.in = data.input.final, 
                 y.total.mse.in = data.label.all,
                 method="ADAM")
print(opt.par$par)

#################################################
library(ggplot2)
library(latex2exp)

plot.length.out = 99
x1.temp = seq(from = -1, to = 5, length.out = plot.length.out)
x2.temp = seq(from = -2, to = 8, length.out = plot.length.out)
data.plot =  expand.grid(x1 = x1.temp, x2 = x2.temp)

# data.plot = data.frame("x1" = data.input.train.final[,1],
#                        "x2" = data.input.train.final[,2])

data.plot$y = (3.956253e-04)*(((-9.526347e-02)*(data.plot$x1+data.plot$x2)^2)+
                             (-3.737216e-01*data.plot$x1 +
                                (1.017811e+00)*data.plot$x2 +
                                (-3.092203e+01)))^2

# mean((data.label.train-data.plot$y)^2)

data.plot$x11 = (-9.526347e-02)*(data.plot$x1+data.plot$x2)^2
print(summary(data.plot$x11))

data.plot$x12 = (-3.737216e-01*data.plot$x1 +
                   (1.017811e+00)*data.plot$x2 +
                   (-3.092203e+01))

print(summary(data.plot$x12))

colnames(data.plot) = c("z1",  "z2",  "y",   "x11", "x12")

x11.temp = seq(from = -16, to = 0, length.out = plot.length.out)
x12.temp = seq(from = -34, to = -22, length.out = plot.length.out)
data.plot.1 =  expand.grid(x11 = x11.temp, x12 = x12.temp)
data.plot.1$y = (3.956253e-04)*(data.plot.1$x11+data.plot.1$x12)^2

png("fig3b_real_y_x11_x12.png", width = 1200, height = 1200)
ggplot.fit = ggplot(data.plot.1, aes(x11, x12, fill= y)) +
  geom_tile()+
  scale_fill_gradient(low = "white",
                      high = "black",
                      guide = "colorbar")+
  labs(title = "") +
  ylab (TeX("$x_{12}$")) +
  xlab(TeX("$x_{11}$")) +
  theme_bw()+
  theme(plot.background = element_rect(fill = "transparent"),
        plot.margin = unit(c(2,1,1,1),units="lines"),
        text = element_text(size=40),
        axis.text.x = element_text(colour="black",size=40,angle=0,hjust=.5,vjust=.5,face="plain"),
        axis.text.y = element_text(colour="black",size=40,angle=0,hjust=1,vjust=0,face="plain"),
        axis.title.x = element_text(colour="black",size=40,angle=0,hjust=.5,vjust=0,face="plain"),
        axis.title.y = element_text(colour="black",size=40,angle=90,hjust=.5,vjust=.5,face="plain"),
        legend.text = element_text(colour="black", size = 40, face = "plain"),
        legend.title = element_text(colour="black", size = 40, face = "plain"),
        legend.key.size = unit(4,"line"),
        legend.position="right", plot.title = element_text(hjust = 0.5),
        legend.box="vertical",
        strip.background = element_blank(),
        strip.placement = "outside",
        panel.spacing = unit(10, "lines"))+
  guides(shape=guide_legend(nrow=1,byrow=TRUE),
         linetype=guide_legend(nrow=1,byrow=TRUE),
         color=guide_legend(nrow=1,byrow=TRUE))
print(ggplot.fit)
dev.off()

png("fig3a_real_y_x1_x2.png", width = 1200, height = 1200)
ggplot.fit = ggplot(data.plot, aes(z1, z2, fill= y)) +
  geom_tile()+
  scale_fill_gradient(low = "white",
                      high = "black",
                      guide = "colorbar")+
  labs(title = "") +
  ylab (TeX("$z_2$")) +
  xlab(TeX("$z_1$")) +
  theme_bw()+
  theme(plot.background = element_rect(fill = "transparent"),
        plot.margin = unit(c(2,1,1,1),units="lines"),
        text = element_text(size=40),
        axis.text.x = element_text(colour="black",size=40,angle=0,hjust=.5,vjust=.5,face="plain"),
        axis.text.y = element_text(colour="black",size=40,angle=0,hjust=1,vjust=0,face="plain"),
        axis.title.x = element_text(colour="black",size=40,angle=0,hjust=.5,vjust=0,face="plain"),
        axis.title.y = element_text(colour="black",size=40,angle=90,hjust=.5,vjust=.5,face="plain"),
        legend.text = element_text(colour="black", size = 40, face = "plain"),
        legend.title = element_text(colour="black", size = 40, face = "plain"),
        legend.key.size = unit(4,"line"),
        legend.position="right", plot.title = element_text(hjust = 0.5),
        legend.box="vertical",
        strip.background = element_blank(),
        strip.placement = "outside",
        panel.spacing = unit(10, "lines"))+
  guides(shape=guide_legend(nrow=1,byrow=TRUE),
         linetype=guide_legend(nrow=1,byrow=TRUE),
         color=guide_legend(nrow=1,byrow=TRUE))
print(ggplot.fit)
dev.off()

png("fig3c_real_x11_x1_x2.png", width = 1200, height = 1200)
ggplot.fit = ggplot(data.plot, aes(z1, z2, fill= x11)) +
  geom_tile()+
  scale_fill_gradient(low = "white",
                      high = "black",
                      guide = "colorbar")+
  labs(title = "") +
  ylab (TeX("$z_2$")) +
  xlab(TeX("$z_1$")) +
  theme_bw()+
  theme(plot.background = element_rect(fill = "transparent"),
        plot.margin = unit(c(2,1,1,1),units="lines"),
        text = element_text(size=40),
        axis.text.x = element_text(colour="black",size=40,angle=0,hjust=.5,vjust=.5,face="plain"),
        axis.text.y = element_text(colour="black",size=40,angle=0,hjust=1,vjust=0,face="plain"),
        axis.title.x = element_text(colour="black",size=40,angle=0,hjust=.5,vjust=0,face="plain"),
        axis.title.y = element_text(colour="black",size=40,angle=90,hjust=.5,vjust=.5,face="plain"),
        legend.text = element_text(colour="black", size = 40, face = "plain"),
        legend.title = element_text(colour="black", size = 40, face = "plain"),
        legend.key.size = unit(4,"line"),
        legend.position="right", plot.title = element_text(hjust = 0.5),
        legend.box="vertical",
        strip.background = element_blank(),
        strip.placement = "outside",
        panel.spacing = unit(10, "lines"))+
  guides(shape=guide_legend(nrow=1,byrow=TRUE),
         linetype=guide_legend(nrow=1,byrow=TRUE),
         color=guide_legend(nrow=1,byrow=TRUE))
print(ggplot.fit)
dev.off()

png("fig3d_real_x12_x1_x2.png", width = 1200, height = 1200)
ggplot.fit = ggplot(data.plot, aes(z1, z2, fill= x12)) +
  geom_tile()+
  scale_fill_gradient(low = "white",
                      high = "black",
                      guide = "colorbar")+
  labs(title = "") +
  ylab (TeX("$z_2$")) +
  xlab(TeX("$z_1$")) +
  theme_bw()+
  theme(plot.background = element_rect(fill = "transparent"),
        plot.margin = unit(c(2,1,1,1),units="lines"),
        text = element_text(size=40),
        axis.text.x = element_text(colour="black",size=40,angle=0,hjust=.5,vjust=.5,face="plain"),
        axis.text.y = element_text(colour="black",size=40,angle=0,hjust=1,vjust=0,face="plain"),
        axis.title.x = element_text(colour="black",size=40,angle=0,hjust=.5,vjust=0,face="plain"),
        axis.title.y = element_text(colour="black",size=40,angle=90,hjust=.5,vjust=.5,face="plain"),
        legend.text = element_text(colour="black", size = 40, face = "plain"),
        legend.title = element_text(colour="black", size = 40, face = "plain"),
        legend.key.size = unit(4,"line"),
        legend.position="right", plot.title = element_text(hjust = 0.5),
        legend.box="vertical",
        strip.background = element_blank(),
        strip.placement = "outside",
        panel.spacing = unit(10, "lines"))+
  guides(shape=guide_legend(nrow=1,byrow=TRUE),
         linetype=guide_legend(nrow=1,byrow=TRUE),
         color=guide_legend(nrow=1,byrow=TRUE))
print(ggplot.fit)
dev.off()




