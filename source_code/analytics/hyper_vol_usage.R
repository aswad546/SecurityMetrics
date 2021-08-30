# MIT License
#
# Copyright (c) 2021, Sohail Habib
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ------------------------------------------------------------------------------------------------------------------------
#

# Title     : Hypervolumes calculations
# Objective : Calculate hyper volume and perform PCA
# Created by: Sohail
# Created on: 2020-07-27

library(dynRB)
library(ggplot2)
library(reshape2)
library(vegan)
library(RColorBrewer)
library(hypervolume)
library(glue)

init_wd <- getwd()
args <- commandArgs(trailingOnly = TRUE)
print('Path=', args[1])
print('File=', args[2])
exp_dir_path <- file.path(args[1])
setwd(exp_dir_path)

data_file_list <- list(args[2])

for (group in data_file_list){
  print(glue("generating results file {group}"))
data_file_name = glue("{group}.csv")
data_path <- file.path(exp_dir_path, data_file_name)

data <- read.csv(data_path, header = TRUE)

hyp_vol_calc <- dynRB_VPa(data, pca.corr = FALSE, correlogram = FALSE, steps = 201)
hyp_vol_size_dim <- dynRB_Vn(data)
hyp_vol_ovrelap_dim <- dynRB_Pn(data)

theme_change <- theme(
plot.background = element_blank(),
panel.grid.minor = element_blank(),
panel.grid.major = element_blank(),
panel.background = element_blank(),
panel.border = element_blank(),
axis.line = element_blank(),
axis.ticks = element_blank(),
axis.text.x = element_text(colour="black", size = rel(1.5), angle=35, hjust = 1),
axis.text.y = element_text(colour="black", size = rel(1.5)),
axis.title.x = element_blank(),
axis.title.y = element_blank()
)
result <- hyp_vol_calc$result
write.csv(result, glue('gr{group}_hyper_vol_overlap_df.csv'))

write.csv(hyp_vol_size_dim$result, glue('gr{group}_hyper_vol_size_per_dim_overlap_df.csv'))

write.csv(hyp_vol_ovrelap_dim$result, glue('gr{group}_hyper_vol_size_overlap_per_dim_overlap_df.csv'))

Overlap_prod <- as.numeric(ifelse(result$V1 == result$V2, "NA", result$port_prod))

# 'result$port_prod' may be changed to 'result$port_mean' or 'result$port_gmean'
is.numeric(Overlap_prod)
Result2<-cbind(result, Overlap_prod)
breaks <- seq(min(Overlap_prod, na.rm=TRUE), max(Overlap_prod, na.rm=TRUE),
              by=round(max(Overlap_prod, na.rm=TRUE)/10, digits=10))
col1 <- colorRampPalette(c("white", "yellow")) #define color gradient
ggplot(Result2, aes(x = V1, y = V2)) +
  geom_tile(data = subset(Result2, !is.na(Overlap_prod)), aes(fill = Overlap_prod), color="black") +
geom_tile(data = subset(Result2, is.na(Overlap_prod)), fill = "lightgrey", color="black") +
  geom_text(aes(label = round(result$port_prod, 2))) +
 scale_fill_gradientn(colours=col1(8), breaks=breaks, guide="colorbar",
limits=c(min(Overlap_prod, na.rm=TRUE), max(Overlap_prod, na.rm=TRUE))) +
theme_change+
  ggsave(glue("gr{group}_overlap_df_prod.png"))


Overlap_mean <- as.numeric(ifelse(result$V1 == result$V2, "NA", result$port_mean))

# 'result$port_prod' may be changed to 'result$port_mean' or 'result$port_gmean'
is.numeric(Overlap_mean)
Result2<-cbind(result, Overlap_mean)
breaks <- seq(min(Overlap_mean, na.rm=TRUE), max(Overlap_mean, na.rm=TRUE),
              by=round(max(Overlap_mean, na.rm=TRUE)/10, digits=10))
col1 <- colorRampPalette(c("white", "yellow")) #define color gradient
ggplot(Result2, aes(x = V1, y = V2)) +
  geom_tile(data = subset(Result2, !is.na(Overlap_mean)), aes(fill = Overlap_mean), color="black") +
geom_tile(data = subset(Result2, is.na(Overlap_mean)), fill = "lightgrey", color="black") +
  geom_text(aes(label = round(result$port_mean, 2))) +
 scale_fill_gradientn(colours=col1(8), breaks=breaks, guide="colorbar",
limits=c(min(Overlap_mean, na.rm=TRUE), max(Overlap_mean, na.rm=TRUE))) +
theme_change+
  ggsave(glue("gr{group}_overlap_df_mean.png"))
}
setwd(init_wd)

