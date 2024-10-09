set.seed(0)

locallyScaledAffinity <- function(dists){
  dists = as.matrix(dists)
  n = nrow(dists)
  scales = apply(dists, 1, function(i) sort(i)[7])
  
  A = matrix(rep(0, length(dists)), nrow=n)
  for (i in 1:n){
    for (j in 1:n){
      if (i != j){
        a_ij = exp(-dists[i,j]/(scales[i]*scales[j]))
        A[i,j] = a_ij
      }
    }
  }
  
  return(A)
}

# data(iris)
# X = iris[,1:4]
# 
# clusters = speccalt(locallyScaledAffinity(dist(X)))
# 
# irisClusters = iris
# irisClusters$Clusters = as.factor(clusters)
# 
# pca = prcomp(X, center=TRUE, retx=TRUE)
# 
library(ggfortify)
# print(autoplot(pca, data=irisClusters, colour="Clusters", shape="Species")  +
#   theme(text=element_text(size=8)))
# 
# 
# library(kernlab)
# data(spirals)
# spiralClusters = as.data.frame(spirals)
# clusters = as.factor(speccalt(locallyScaledAffinity(dist(spirals))))
# spiralClusters$Clusters = as.factor(clusters)
# 
# # print(autoplot(spiralClusters, colour=clusters)  +
# #         theme(text=element_text(size=8)))
# 
# print(ggplot(data = spiralClusters, aes(V1, V2, color = Clusters)) +
#   geom_point())


X1 = cbind(rnorm(100), rnorm(100))
X2 = matrix(rep(0, 200), nrow=100)
for (i in 1:100){
  X2[i,] = c(cos(2*pi*i/100), sin(2*pi*i/100)) * 8
}
X3_x = rep(seq(-25,25,5), 11)
X3_y = rep(-25,11)
for (i in seq(-20,25,5)){
  X3_y = c(X3_y, rep(i, 11))
}
X3 = cbind(X3_x, X3_y)
# X3 = matrix(rep(0, 450), nrow=225)
# for (i in 1:100){
#   X3[i] = -7:7
# }

X = rbind(X1, X2, X3)
plot(X)

# XClusters = as.data.frame(X)
# clusters = as.factor(speccalt(locallyScaledAffinity(dist(X))))
# XClusters$Clusters = as.factor(clusters)
# 
# print(ggplot(data = XClusters, aes(V1, V2, color = Clusters)) +
#         geom_point())

library(Spectrum)

XClusters = as.data.frame(t(X))
clusters = as.factor(Spectrum(XClusters, method=2)$assignments)
XClusters = as.data.frame(X)
XClusters$Clusters = as.factor(clusters)
colnames(XClusters) <- c("X", "Y", "Clusters")

print(ggplot(data = XClusters, aes(X, Y, color = Clusters)) +
        geom_point()) + ggtitle("Self-Tuning Spectral Clustering")

# XClusters$Clusters = as.factor(kmeans(X, 3, algorithm="Lloyd"))
# print(ggplot(data = XClusters, aes(X, Y, color = Clusters)) +
#         geom_point())

med = median(dist(X))
sc = specc(X, centers=3, kernel="rbfdot", kpar=list(sigma=med))
XClusters$Clusters = as.factor(sc)
print(ggplot(data = XClusters, aes(X, Y, color = Clusters)) +
        geom_point()) + ggtitle("Regular Spectral Clustering With Median Trick")

