# y_A and y_B are vectors of evals for two models
import numpy as np

S_A, S_B = y_A.sum(), y_B.sum()
# draw posterior samples (ps)
ps_A = np.random.beta(1 + S_A, 1 + (N - S_A), size=2000)
ps_B = np.random.beta(1 + S_B, 1 + (N - S_B), size=2000)
# posterior difference and 95% QBI
ps_diff = ps_A - ps_B  
bayes_diff = np.percentile(ps_diff, [2.5, 97.5]) 
# posterior odds ratio and 95% QBI
ps_or = (ps_A / (1 - ps_A)) / (ps_B / (1 - ps_B))
bayes_or = np.percentile(ps_or, [2.5, 97.5]) 