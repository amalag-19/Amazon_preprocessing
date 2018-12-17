########################################################################################################
## Loading the required packages
library(ggplot2)

########################################################################################################
########################################################################################################
## Sourcing the all code
source(file = "/Users/Amal/Box Sync/PSU/Fall 2018/Main_Research/Network Models/Project 5 (Bipartite)/code/bipartite_All/bipartite_All.R")

########################################################################################################
## Running the Amazon data
load(file = "/Users/Amal/Box Sync/PSU/Fall 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/amazon_5core/2014/subsets/1-5core/amazon_adjacency.RData")

load(file = "/Users/Amal/Box Sync/PSU/Fall 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/amazon_5core/2014/subsets/1-5core/amazon_rating.RData")

load(file = "/Users/Amal/Box Sync/PSU/Fall 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/amazon_5core/2014/subsets/1-5core/df_reviewer_2014_1_5_core.RData")

load(file = "/Users/Amal/Box Sync/PSU/Fall 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/amazon_5core/2014/subsets/1-5core/df_product_2014_1_5_core.RData")

N1<-dim(amazon_adjacency)[1]
N2<-dim(amazon_adjacency)[2]
#undebug(wrapper_modsel)
#net_result<-wrapper_modsel(K1 = 2, K2 = 2)
########################################################################################################
## Defining all possible cluster pairs
# clust_mat<-matrix(c(2,2,2,3,3,2,3,3,2,4,4,2,3,4,4,3,4,4),9,2,byrow = T)
# 
# net_result<-list()
# for(i in 1:nrow(clust_mat)){
#   theta_init<-jitter(matrix(rep(0,clust_mat[i,1]*clust_mat[i,2]),clust_mat[i,1],clust_mat[i,2]))
#   omega_init<-jitter(matrix(rep(0,clust_mat[i,1]*clust_mat[i,2]),clust_mat[i,1],clust_mat[i,2]))
#   mu_init_sub<-c(-0.5+jitter(rep(0,clust_mat[i,1]*clust_mat[i,2])),jitter(rep(0,clust_mat[i,1]*clust_mat[i,2])),0.5+jitter(rep(0,clust_mat[i,1]*clust_mat[i,2])))
#   mu_init <- array(c(rep(-1,clust_mat[i,1]*clust_mat[i,2]),mu_init_sub,rep(1,clust_mat[i,1]*clust_mat[i,2])),dim=c(clust_mat[i,1],clust_mat[i,2],5))
#   net_result[[i]]<-wrapper_bipartite_model1(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = clust_mat[i,1], nclust2 = clust_mat[i,2], thres = 10^(-6),theta_init = theta_init, omega_init = omega_init, mu_init = mu_init,sim_indicator = 0,R = 5)
# }
# 
# str(net_result)


########################################################################################################
theta_init<-jitter(matrix(rep(0,9),3,3))
omega_init<-jitter(matrix(rep(0,9),3,3))
mu_init_sub<-c(-0.5+jitter(rep(0,9)),jitter(rep(0,9)),0.5+jitter(rep(0,9)))
mu_init <- array(c(rep(-1,9),mu_init_sub,rep(1,9)),dim=c(3,3,5))

# ## Running the banchmark model 1
# net_result<-wrapper_bipartite_model1(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, omega_init = omega_init,  mu_init = mu_init, sim_indicator = 0,R = 5)
# 
# save(net_result,file="/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_2014_10core/fitted_model_results/base_model/base_model.RData")
# beta_u_init<-jitter(matrix(rep(0,3*2),3,2))
# beta_p_init<-jitter(matrix(rep(0,3*2),3,2))
# delta_u_init<-jitter(matrix(rep(0,3*2),3,2))
# delta_p_init<-jitter(matrix(rep(0,3*2),3,2))
# 
# undebug(wrapper_bipartite_model2)
# net_result<-wrapper_bipartite_model2(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u = as.matrix(cbind((df_reviewer_2014_10core[,"num_product"]/1749),(df_reviewer_2014_10core[,"mean_rating"]/5))), cov_p = as.matrix(cbind((df_product_2014_10core[,"num_reviewer"]/821),(df_product_2014_10core[,"mean_rating"]/5))), sim_indicator = 0,R = 5)

## Running the covariate model 1
C1<-1
C2<-1

beta_u_init<-jitter(matrix(rep(0,3*C1),3,C1))
beta_p_init<-jitter(matrix(rep(0,3*C2),3,C2))
delta_u_init<-jitter(matrix(rep(0,3*C1),3,C1))
delta_p_init<-jitter(matrix(rep(0,3*C2),3,C2))

#net_result<-wrapper_bipartite_model2(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u = matrix(df_fair[,"Score"],N1,1), cov_p = matrix(df_good[,"Score"],N2,1), sim_indicator = 0,R = 5)

# net_result<-wrapper_bipartite_model2(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u = matrix(cbind(df_fair[,"Score"],(df_fair[,"num_product"]/N2)),N1,2), cov_p = matrix(cbind(df_good[,"Score"],(df_good[,"num_reviewer"]/N1)),N2,2), sim_indicator = 0,R = 5)

net_result<-wrapper_bipartite_model2(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u =matrix((df_reviewer_2014_1_5_core[,"num_product"]/N2),N1,1), cov_p = matrix((df_product_2014_1_5_core[,"num_reviewer"]/N1),N2,1), sim_indicator = 0,R = 5)

save(net_result,file="/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_2014_10core/fitted_model_results/num_rev_cov/")

str(net_result)

## theta
net_result[[1]][[5]]

## beta_u
net_result[[1]][[6]]

## beta_p
net_result[[1]][[7]]

## omega
net_result[[1]][[8]]

## mu
net_result[[1]][[9]]

## delta_u
net_result[[1]][[10]]

## delta_p
net_result[[1]][[11]]

########################################################################################################
########################################################################################################
theta_init<-jitter(matrix(rep(0,9),3,3))
omega_init<-jitter(matrix(rep(0,9),3,3))
mu_init_sub<-c(-0.5+jitter(rep(0,9)),jitter(rep(0,9)),0.5+jitter(rep(0,9)))
mu_init <- array(c(rep(-1,9),mu_init_sub,rep(1,9)),dim=c(3,3,5))

C1_outer<-1
C2_outer<-1
C1_ordistic<-1
C2_ordistic<-1

beta_u_init<-jitter(matrix(rep(0,3*C1_outer),3,C1_outer))
beta_p_init<-jitter(matrix(rep(0,3*C2_outer),3,C2_outer))
delta_u_init<-jitter(matrix(rep(0,3*C1_ordistic),3,C1_ordistic))
delta_p_init<-jitter(matrix(rep(0,3*C2_ordistic),3,C2_ordistic))

## does not work
net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = matrix((df_fair[,"num_product"]/N2),N1,1), cov_p_outer = matrix((df_good[,"num_reviewer"]/N1),N2,1), cov_u_ordistic = matrix(df_fair[,"Score"],N1,1), cov_p_ordistic = matrix(df_good[,"Score"],N2,1), sim_indicator = 0,R = 5)

## does not work
num_prod_std<-(df_fair[,"num_product"]-mean(df_fair[,"num_product"]))/sqrt(var(df_fair[,"num_product"]))
num_rev_std<-(df_good[,"num_reviewer"]-mean(df_good[,"num_reviewer"]))/sqrt(var(df_good[,"num_reviewer"]))

net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = matrix(num_prod_std,N1,1), cov_p_outer = matrix(num_rev_std,N2,1), cov_u_ordistic = matrix(df_fair[,"mean_rating"],N1,1), cov_p_ordistic = matrix(df_good[,"mean_rating"],N2,1), sim_indicator = 0,R = 5)

## does not work
net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = matrix((df_fair[,"num_product"]/N2),N1,1), cov_p_outer = matrix((df_good[,"num_reviewer"]/N1),N2,1), cov_u_ordistic = matrix(df_fair[,"median_rating"],N1,1), cov_p_ordistic = matrix(df_good[,"median_rating"],N2,1), sim_indicator = 0,R = 5)

## does not work
net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = matrix(df_fair[,"Score"],N1,1), cov_p_outer = matrix(df_good[,"Score"],N2,1), cov_u_ordistic = matrix(df_fair[,"mean_rating"],N1,1), cov_p_ordistic = matrix(df_good[,"mean_rating"],N2,1), sim_indicator = 0,R = 5)

## trying
net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = matrix(df_fair[,"Score"],N1,1), cov_p_outer = matrix(df_good[,"Score"],N2,1), cov_u_ordistic = matrix((df_fair[,"num_product"]/N2),N1,1), cov_p_ordistic = matrix((df_good[,"num_reviewer"]/N1),N2,1), sim_indicator = 0,R = 5)



net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = matrix(df_fair[,"Score"],N1,1), cov_p_outer = matrix(df_good[,"Score"],N2,1), cov_u_ordistic = matrix(num_prod_std,N1,1), cov_p_ordistic = matrix(num_rev_std,N2,1), sim_indicator = 0,R = 5)

net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = matrix(df_fair[,"Score"],N1,1), cov_p_outer = matrix(df_good[,"Score"],N2,1), cov_u_ordistic = matrix((df_fair[,"num_product"]),N1,1), cov_p_ordistic = matrix((df_good[,"num_reviewer"]),N2,1), sim_indicator = 0,R = 5)

save(net_result,file="/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_2014_10core/fitted_model_results/outer_fair_good_ordistic_rev.RData")

net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = matrix(df_fair[,"Score"],N1,1), cov_p_outer = matrix(df_good[,"Score"],N2,1), cov_u_ordistic = matrix((num_prod_std),N1,1), cov_p_ordistic = matrix((num_rev_std),N2,1), sim_indicator = 0,R = 5)

save(net_result,file="/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_2014_10core/fitted_model_results/outer_fair_good_ordistic_stdrev.RData")

##################################################
C1_outer<-1
C2_outer<-1
C1_ordistic<-1
C2_ordistic<-1

beta_u_init<-jitter(matrix(rep(0,3*C1_outer),3,C1_outer))
beta_p_init<-jitter(matrix(rep(0,3*C2_outer),3,C2_outer))
delta_u_init<-jitter(matrix(rep(0,3*C1_ordistic),3,C1_ordistic))
delta_p_init<-jitter(matrix(rep(0,3*C2_ordistic),3,C2_ordistic))

#net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = as.matrix(df_reviewer_2014_1_5_core[,c("num_product","mean_rating","fairness")]), cov_p_outer = as.matrix(df_product_2014_1_5_core[,c("num_reviewer","mean_rating","goodness","price","salesRank")]), cov_u_ordistic = as.matrix(df_reviewer_2014_1_5_core[,c("num_product","mean_rating","fairness")]), cov_p_ordistic = as.matrix(df_product_2014_1_5_core[,c("num_reviewer","mean_rating","goodness","price","salesRank")]), sim_indicator = 0,R = 5)

#net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = as.matrix(df_reviewer_2014_1_5_core[,c("num_product","mean_rating","fairness")]), cov_p_outer = as.matrix(df_product_2014_1_5_core[,c("num_reviewer","mean_rating","goodness","price","salesRank")]), cov_u_ordistic = as.matrix(df_reviewer_2014_1_5_core[,c("num_product")]), cov_p_ordistic = as.matrix(df_product_2014_1_5_core[,c("num_reviewer")]), sim_indicator = 0,R = 5)

#net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = as.matrix(df_reviewer_2014_1_5_core[,c("num_product","fairness")]), cov_p_outer = as.matrix(df_product_2014_1_5_core[,c("num_reviewer","goodness","price","salesRank")]), cov_u_ordistic = as.matrix(df_reviewer_2014_1_5_core[,c("num_product")]), cov_p_ordistic = as.matrix(df_product_2014_1_5_core[,c("num_reviewer")]), sim_indicator = 0,R = 5)

net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = as.matrix(df_reviewer_2014_1_5_core[,c("fairness")]), cov_p_outer = as.matrix(df_product_2014_1_5_core[,c("goodness")]), cov_u_ordistic = as.matrix(df_reviewer_2014_1_5_core[,c("num_product")]), cov_p_ordistic = as.matrix(df_product_2014_1_5_core[,c("num_reviewer")]), sim_indicator = 0,R = 5)

net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = as.matrix(df_reviewer_2014_1_5_core[,c("fairness")]), cov_p_outer = as.matrix(df_product_2014_1_5_core[,c("goodness")]), cov_u_ordistic = as.matrix(df_reviewer_2014_1_5_core[,c("fairness")]), cov_p_ordistic = as.matrix(df_product_2014_1_5_core[,c("goodness")]), sim_indicator = 0,R = 5)

str(net_result)


net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = as.matrix(df_reviewer_2014_1_5_core[,c("num_product","fairness")]), cov_p_outer = as.matrix(df_product_2014_1_5_core[,c("num_reviewer","goodness")]), cov_u_ordistic = as.matrix(df_reviewer_2014_1_5_core[,c("num_product")]), cov_p_ordistic = as.matrix(df_product_2014_1_5_core[,c("num_reviewer")]), sim_indicator = 0,R = 5)

debug(wrapper_bipartite_model3)
net_result<-wrapper_bipartite_model3(net_adjacency = amazon_adjacency, net_rating = amazon_rating, nclust1 = 3, nclust2 = 3, thres = 10^(-6),theta_init = theta_init, beta_u_init = beta_u_init, beta_p_init = beta_p_init, omega_init = omega_init,  mu_init = mu_init, delta_u_init = delta_u_init, delta_p_init = delta_p_init, cov_u_outer = as.matrix(df_reviewer_2014_1_5_core[,c("num_product","fairness")]), cov_p_outer = as.matrix(df_product_2014_1_5_core[,c("num_reviewer","goodness")]), cov_u_ordistic = as.matrix(df_reviewer_2014_1_5_core[,c("num_product","fairness")]), cov_p_ordistic = as.matrix(df_product_2014_1_5_core[,c("num_reviewer","goodness")]), sim_indicator = 0,R = 5)


str(net_result)

########################################################################################################
########################################################################################################
## Ratings plot for K1*K2 blocks

load(file = "/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_2014_10core/fitted_model_results/outer_fair_good_ordistic_rev.RData")
block_list<-list()

K1<-3
K2<-3

for(k in 1:K1){
  block_list[[k]]<-list()
  for(l in 1:K2){
    ##initializing the ratings vector in k,l block
    rating_vec<-c()
    for (i in 1:N1){
      for (j in 1:N2){
        if(amazon_adjacency[i,j]==1){
          if((net_result[["Set1_estimated_cluster_IDs"]][i]==k)&(net_result[["Set2_estimated_cluster_IDs"]][j]==l)){
            rating_vec<-c(rating_vec,amazon_rating[i,j])
          }
        }
      }
      #print(i)
    }
    block_list[[k]][[l]]<-rating_vec
    print(k)
    print(l)
  }
}

length(block_list[[1]][[1]])
length(block_list[[1]][[2]])
length(block_list[[1]][[3]])
length(block_list[[2]][[1]])
length(block_list[[2]][[2]])
length(block_list[[2]][[3]])
length(block_list[[3]][[1]])
length(block_list[[3]][[2]])
length(block_list[[3]][[3]])



df_rating<-data.frame("rev_clust"=numeric(),"prod_clust"=numeric(),"rating"=numeric())

row_counter<-1
block_count_mat<-matrix(NA_real_,K1,K2)
for(k in 1:K1){
  for(l in 1:K2){
    n<-length(block_list[[k]][[l]])
    block_count_mat[k,l]<-n
    if(n>0){
      df_rating[row_counter:(row_counter+(n-1)),"rev_clust"]<-rep(paste0("Rev.Seg.",k),n)
      df_rating[row_counter:(row_counter+(n-1)),"prod_clust"]<-rep(paste0("Prod.Seg.",l),n)
      df_rating[row_counter:(row_counter+(n-1)),"rating"]<-block_list[[k]][[l]]
      row_counter<-row_counter+n
    }
  }
}

p<-ggplot(data = df_rating)

p+geom_histogram(mapping = aes(x = factor(rating)),stat = "count",binwidth = 500)+facet_grid(factor(rev_clust)~factor(prod_clust))+theme_bw()+labs(x="Rating",y="Count")+theme(axis.text.x = element_text(size=20),axis.text.y = element_text(size=20),axis.title.x=element_text(size=20),axis.title.y=element_text(size=20),strip.text = element_text(size=15))

########################################################################################################
## Summary of covariates

summary(df_fair$num_product)[c(1,3,4,6)]
sqrt(var(df_fair$num_product))

summary(df_good$num_reviewer)[c(1,3,4,6)]
sqrt(var(df_good$num_reviewer))

summary(df_fair$Score)[c(1,3,4,6)]
sqrt(var(df_fair$Score))

summary(df_good$Score)[c(1,3,4,6)]
sqrt(var(df_good$Score))

summary(df_fair$mean_rating)[c(1,3,4,6)]
sqrt(var(df_fair$mean_rating))

summary(df_good$mean_rating)[c(1,3,4,6)]
sqrt(var(df_good$mean_rating))

row.names(df_fair)<-NULL

######################################################
head(df_fair)
sum(net_result[["Set2_estimated_cluster_IDs"]]==3)

df_fair$Cluster_ID<-net_result[["Set1_estimated_cluster_IDs"]]
head(df_fair)

summary(df_fair[which(df_fair$Cluster_ID==1),"num_product"])[c(1,4,3,6)]
sqrt(var(df_fair[which(df_fair$Cluster_ID==1),"num_product"]))

summary(df_fair[which(df_fair$Cluster_ID==2),"num_product"])[c(1,4,3,6)]
sqrt(var(df_fair[which(df_fair$Cluster_ID==2),"num_product"]))

summary(df_fair[which(df_fair$Cluster_ID==3),"num_product"])[c(1,4,3,6)]
sqrt(var(df_fair[which(df_fair$Cluster_ID==3),"num_product"]))

######################################################
summary(df_fair[which(df_fair$Cluster_ID==1),"Score"])[c(1,4,3,6)]
sqrt(var(df_fair[which(df_fair$Cluster_ID==1),"Score"]))

summary(df_fair[which(df_fair$Cluster_ID==2),"Score"])[c(1,4,3,6)]
sqrt(var(df_fair[which(df_fair$Cluster_ID==2),"Score"]))

summary(df_fair[which(df_fair$Cluster_ID==3),"Score"])[c(1,4,3,6)]
sqrt(var(df_fair[which(df_fair$Cluster_ID==3),"Score"]))

######################################################
summary(df_fair[which(df_fair$Cluster_ID==1),"mean_rating"])[c(1,4,3,6)]
sqrt(var(df_fair[which(df_fair$Cluster_ID==1),"mean_rating"]))

summary(df_fair[which(df_fair$Cluster_ID==2),"mean_rating"])[c(1,4,3,6)]
sqrt(var(df_fair[which(df_fair$Cluster_ID==2),"mean_rating"]))

summary(df_fair[which(df_fair$Cluster_ID==3),"mean_rating"])[c(1,4,3,6)]
sqrt(var(df_fair[which(df_fair$Cluster_ID==3),"mean_rating"]))

######################################################
######################################################
row.names(df_good)<-NULL

head(df_good)
df_good$Cluster_ID<-net_result[["Set2_estimated_cluster_IDs"]]
head(df_good)

write.csv(df_good[,c("asin","Cluster_ID")],file = "/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/amazon_2014_10core/product_segmented_IDs.csv")

summary(df_good[which(df_good$Cluster_ID==1),"num_reviewer"])[c(1,4,3,6)]
sqrt(var(df_good[which(df_good$Cluster_ID==1),"num_reviewer"]))

summary(df_good[which(df_good$Cluster_ID==2),"num_reviewer"])[c(1,4,3,6)]
sqrt(var(df_good[which(df_good$Cluster_ID==2),"num_reviewer"]))

summary(df_good[which(df_good$Cluster_ID==3),"num_reviewer"])[c(1,4,3,6)]
sqrt(var(df_good[which(df_good$Cluster_ID==3),"num_reviewer"]))

######################################################
summary(df_good[which(df_good$Cluster_ID==1),"Score"])[c(1,4,3,6)]
sqrt(var(df_good[which(df_good$Cluster_ID==1),"Score"]))

summary(df_good[which(df_good$Cluster_ID==2),"Score"])[c(1,4,3,6)]
sqrt(var(df_good[which(df_good$Cluster_ID==2),"Score"]))

summary(df_good[which(df_good$Cluster_ID==3),"Score"])[c(1,4,3,6)]
sqrt(var(df_good[which(df_good$Cluster_ID==3),"Score"]))

######################################################
summary(df_good[which(df_good$Cluster_ID==1),"mean_rating"])[c(1,4,3,6)]
sqrt(var(df_good[which(df_good$Cluster_ID==1),"mean_rating"]))

summary(df_good[which(df_good$Cluster_ID==2),"mean_rating"])[c(1,4,3,6)]
sqrt(var(df_good[which(df_good$Cluster_ID==2),"mean_rating"]))

summary(df_good[which(df_good$Cluster_ID==3),"mean_rating"])[c(1,4,3,6)]
sqrt(var(df_good[which(df_good$Cluster_ID==3),"mean_rating"]))

########################################################################################################

rating_vec<-c()
for (i in 1:N1){
  for (j in 1:N2){
    if(amazon_adjacency[i,j]==1){
      rating_vec<-c(rating_vec,amazon_rating[i,j])
    }
  }
}
df_rating_overall<-data.frame("rating"=rating_vec)
p<-ggplot(data = df_rating_overall)

p+geom_histogram(mapping = aes(x = factor(rating)),stat = "count",binwidth = 500)+theme_bw()+labs(x="Rating",y="Count")+theme(axis.text.x = element_text(size=30),axis.text.y = element_text(size=30),axis.title.x=element_text(size=30),axis.title.y=element_text(size=30))+scale_y_continuous(breaks = round(seq(0, 3000, by = 300),1))



head(df_good)

########################################################################################################
## getting row ids for asins

df_asins<-read.csv(file = "/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/bipartite network/all_asins.csv",header = F)

head(df_asins)
nrow(df_asins)

row_IDs_vec<-rep(NA_integer_,nrow(df_good))
for(i in 1:nrow(df_good)){
  row_IDs_vec[i]<-which(df_asins$V1==df_good$asin[i])
  if(i%%100==0){
    print(i)
  }
}

write.csv(row_IDs_vec,file = "/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/bipartite network/asins_row_IDs.csv")


########################################################################################################
library(stringr)

# Reading categories info.
df_categories<-read.csv(file = "/Users/Amal/Box Sync/PSU/Summer 2018/Main_Research/Network Models/Project 5 (Bipartite)/Real Data/bipartite network/result_categories.csv",header = F,stringsAsFactors = F)

categories_vec<-rep(NA_character_,nrow(df_categories))
for (i in 1:nrow(df_categories)){
  categories_vec[i]<-str_c(df_categories[i,], collapse = ",")
}

str(categories_vec)
df_good$category<-rep(NA_character_,nrow(df_good))

clothing_count<-0
shoe_count<-0
jewelry_count<-0
others_count<-0
count_vec<-rep(0,nrow(df_good))
for(i in 1:nrow(df_good)){
  indicator_mat<-matrix(0,ncol(df_categories),4)
  for (j in 1:ncol(df_categories)){
    if(str_count(df_categories[i,j], "Clothing")>=2){
      df_good$category[i]<-"Clothing"
      clothing_count<-clothing_count+1
      indicator_mat[j,1]<-1
    }
    if(str_count(df_categories[i,j], "Shoes")>=2){
      df_good$category[i]<-"Shoes"
      shoe_count<-shoe_count+1
      indicator_mat[j,2]<-1
    }
    if(str_count(df_categories[i,j], "Jewelry")>=2){
      df_good$category[i]<-"Jewelry"
      jewelry_count<-jewelry_count+1
      indicator_mat[j,3]<-1
    }
    if(is.na(df_good$category[i])){
      df_good$category[i]<-"Others"
      others_count<-others_count+1
      indicator_mat[j,4]<-1
    }
  }
  if(any(as.vector(apply(X = indicator_mat,MARGIN = 1,sum))>1)){
    count_vec[i]<-count_vec[i]+1
  }
}

max(count_vec)

df_good$category[132]

str_count(categories_vec[1], "Shoes")

df_prod<-data.frame("Cluster_ID"=net_result[["Set2_estimated_cluster_IDs"]],"Category"=df_good[,"category"],"Count"=rep(1,nrow(df_good)),stringsAsFactors = F)

str(df_prod)

library(plyr)
df_prod_modified<-ddply(df_prod,.(Cluster_ID,Category),summarise,Count = sum(Count))

df_prod_modified2<-cbind(ddply(df_prod_modified,.(Cluster_ID),summarise,Percentage = 100*Count/1749),"Category"=df_prod_modified$Category)

p<-ggplot(data = df_prod_modified2,aes(x = factor(Cluster_ID),y=Percentage,fill=factor(Category,levels=c("Clothing","Shoes","Jewelry","Others"))))

p+geom_bar(stat = "identity")+theme_bw()+labs(x="Product Segmentation ID", y="Percentage")+theme(axis.text.x = element_text(size=30),axis.text.y = element_text(size=30),legend.text=element_text(size=20),strip.text.x = element_text(size = 30),axis.title.x=element_text(size=30),axis.title.y=element_text(size=30),legend.title = element_text(hjust = 0.5,size = 20))+scale_fill_discrete(guide = guide_legend(title = "Category"))

########################################################################################################

load(file = "/Users/Amal/Box Sync/PSU/Fall 2018/Main_Research/Network Models/Project 5 (Bipartite)/code/Amazon_Data_Analysis/Amazon_Cluster_Code/benchmark_model1_cluster_code/net_result_model_selection.RData")

str(net_result_model_selection)

## Initializing the cluster pair matrix
clust_mat<-matrix(c(2,2,2,3,3,2,3,3,2,4,4,2,3,4,4,3,4,4,2,5,5,2,3,5,5,3,4,5,5,4,5,5),16,2,byrow = T)

ICL_vec<-rep(NA_real_,nrow(clust_mat))

N1<-1000
N2<-1000
R<-5

for(i in 1:length(net_result_model_selection)){
  param_converge<-net_result_model_selection[[i]][[1]]
  Gvals_cube_0<-get_Gval_0(omega=param_converge[[6]], mu=param_converge[[7]],K1=clust_mat[i,1],K2=clust_mat[i,2],R=R)
  Gvals_mat_0=apply(X = Gvals_cube_0,MARGIN = c(1,2),sum)
  ICL_vec[i]<-ICL_cal(cluster_ids_est_U = net_result_model_selection[[i]][[2]], cluster_ids_est_P = net_result_model_selection[[i]][[3]], pi_U=param_converge[[3]], pi_P=param_converge[[4]] ,theta = param_converge[[5]], Gvals_cube_0 = Gvals_cube_0, Gvals_mat_0 = Gvals_mat_0, net_adjacency=amazon_adjacency, net_rating=amazon_rating, N1 = N1, N2 = N2, K1 = clust_mat[i,1], K2 = clust_mat[i,2], R = 5)
  net_result_model_selection[[i]]$ICL<-ICL_vec[i]
  print(i)
}

str(net_result_model_selection)

which.max(ICL_vec)

########################################################################################################
load(file = "/Users/Amal/Box Sync/PSU/Fall 2018/Main_Research/Network Models/Project 5 (Bipartite)/code/Amazon_Data_Analysis/Amazon_Cluster_Code/final_model/outer/results/param33.RData")
str(net_result)

net_result[[1]][[5]]
net_result[[1]][[6]]
net_result[[1]][[7]]
net_result[[1]][[8]]

net_result[[1]][[9]]
