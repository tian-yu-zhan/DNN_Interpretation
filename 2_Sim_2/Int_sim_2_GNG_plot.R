
rm(list = ls())

setwd("~/Dropbox/Research/AbbVie/Interpretation/3_final_v3/2_Sim_2/")

library(ggplot2)
library(latex2exp)

data.1 = read.csv("sim_2_ori.csv")
data.2 = read.csv("sim_2_int.csv")

data.plot = rbind(data.1[1:18,], data.2[1:18,])
data.plot$id = rep(1:18, 2)
data.plot$grp = factor(rep(1:2, each = 18))


png("sim_2_plot_mcp.png", width = 2400, height = 1200)
ggplot.fit = ggplot(data.plot, aes(x = id, y = cp.val, 
                                   label = p, group = grp)) +
  geom_point(size = 12, aes(shape = grp), stroke = 5)+
  geom_text(color="black", hjust = -1.2, vjust = 0.8, size = 15)+
  # scale_size_manual(values=c(10, 4, 4, 3))+
  scale_shape_manual(values=c(2, 17))+
  # scale_color_manual(values=c("black", "blue", "blue", "black"))+
  # geom_point(aes(x=knots,y=0), color = "blue",
  #            data=data.frame("knots"=knots[c(1:6,12:16)]),
  #            size = 16)+
  # geom_point(aes(x=knots,y=0), color = "blue",
  #            data=data.frame("knots"=knots[7:11]),
  #            size = 16)+
  # scale_color_manual(values=c("blue", "red"))+
  # geom_hline(yintercept=min(data.plot$cp.val)+cp.tol, 
  #            linetype="dashed", color = "black", linewidth = 3) + 
  # scale_linetype_manual(values=c("solid", "longdash", "solid", "longdash", "solid", "solid")) +
  # # scale_alpha_manual(values=c(1, 1, rep(0.7, 5)))+
  # scale_color_manual(values=
  #                      c("black", "#56B4E9", "#0072B2", "#E69F00", "#D55E00", "#009E73"))+
  scale_y_continuous(breaks = c(0, 30, 60, 90), 
                     limits = c(0, 90)) +
  # # scale_y_continuous(sec.axis = sec_axis(~., name = "Type I error")) +
  scale_x_continuous(breaks = 1:18)+
  labs(title = "") +
  ylab (TeX("$MC'_{cv}$")) +
  xlab("Model ID") +
  # theme_bw()+
  theme(plot.background = element_rect(fill = "transparent"),
        plot.margin = unit(c(2,1,1,1),units="lines"),
        text = element_text(size=50),
        axis.text.x = element_text(colour="black",size=60,angle=0,hjust=.5,vjust=.5,face="plain"),
        axis.text.y = element_text(colour="black",size=60,angle=0,hjust=1,vjust=0,face="plain"),
        axis.title.x = element_text(colour="black",size=60,angle=0,hjust=.5,vjust=0,face="plain"),
        axis.title.y = element_text(colour="black",size=60,angle=90,hjust=.5,vjust=.5,face="plain"),
        legend.text = element_text(colour="black", size = 37, face = "plain"),
        legend.title = element_text(colour="black", size = 42, face = "plain"),
        legend.key.size = unit(10,"line"),
        legend.position="none", plot.title = element_text(hjust = 0.5),
        legend.box="vertical",
        strip.background = element_blank(),
        strip.placement = "outside",
        panel.spacing = unit(10, "lines"))+
  guides(shape=guide_legend(nrow=1,byrow=TRUE),
         linetype=guide_legend(nrow=1,byrow=TRUE),
         color=guide_legend(nrow=1,byrow=TRUE))

print(ggplot.fit)
dev.off()


png("sim_2_plot_mse.png", width = 2400, height = 1200)
ggplot.fit = ggplot(data.plot, aes(x = id, y = MSE.val, 
                                   label = p, group = grp)) +
  geom_point(size = 12, aes(shape = grp), stroke = 5)+
  geom_text(color="black", hjust = -1.2, vjust = 0.8, size = 15)+
  # scale_size_manual(values=c(10, 4, 4, 3))+
  scale_shape_manual(values=c(2, 17))+
  # scale_color_manual(values=c("black", "blue", "blue", "black"))+
  # geom_point(aes(x=knots,y=0), color = "blue",
  #            data=data.frame("knots"=knots[c(1:6,12:16)]),
  #            size = 16)+
  # geom_point(aes(x=knots,y=0), color = "blue",
  #            data=data.frame("knots"=knots[7:11]),
  #            size = 16)+
  # scale_color_manual(values=c("blue", "red"))+
  # geom_hline(yintercept=min(data.plot$cp.val)+cp.tol, 
  #            linetype="dashed", color = "black", linewidth = 3) + 
  # scale_linetype_manual(values=c("solid", "longdash", "solid", "longdash", "solid", "solid")) +
# # scale_alpha_manual(values=c(1, 1, rep(0.7, 5)))+
# scale_color_manual(values=
#                      c("black", "#56B4E9", "#0072B2", "#E69F00", "#D55E00", "#009E73"))+
scale_y_continuous(breaks = c(0, 0.04, 0.08, 0.12, 0.16), 
                   limits = c(0, 0.16)) +
  # # scale_y_continuous(sec.axis = sec_axis(~., name = "Type I error")) +
  scale_x_continuous(breaks = 1:18)+
  labs(title = "") +
  ylab (TeX("MSE$'_{cv}$")) +
  xlab("Model ID") +
  # theme_bw()+
  theme(plot.background = element_rect(fill = "transparent"),
        plot.margin = unit(c(2,1,1,1),units="lines"),
        text = element_text(size=50),
        axis.text.x = element_text(colour="black",size=60,angle=0,hjust=.5,vjust=.5,face="plain"),
        axis.text.y = element_text(colour="black",size=60,angle=0,hjust=1,vjust=0,face="plain"),
        axis.title.x = element_text(colour="black",size=60,angle=0,hjust=.5,vjust=0,face="plain"),
        axis.title.y = element_text(colour="black",size=60,angle=90,hjust=.5,vjust=.5,face="plain"),
        legend.text = element_text(colour="black", size = 37, face = "plain"),
        legend.title = element_text(colour="black", size = 42, face = "plain"),
        legend.key.size = unit(10,"line"),
        legend.position="none", plot.title = element_text(hjust = 0.5),
        legend.box="vertical",
        strip.background = element_blank(),
        strip.placement = "outside",
        panel.spacing = unit(10, "lines"))+
  guides(shape=guide_legend(nrow=1,byrow=TRUE),
         linetype=guide_legend(nrow=1,byrow=TRUE),
         color=guide_legend(nrow=1,byrow=TRUE))

print(ggplot.fit)
dev.off()


