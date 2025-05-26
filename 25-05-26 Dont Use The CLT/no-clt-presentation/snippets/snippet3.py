# confusion_arr is np.array([N_TP,N_FP,N_FN,N_TN])
from numpy.random import dirichlet
from arviz import hdi

ps = dirichlet(confusion_arr + 1, 2000) 
f1_samples = calculate_f1(ps) # implements Eq.10
# 95% HDI and QBI
bayes_hdi = hdi(f1_samples, hdi_prob=0.95)
bayes_qbi = np.percentile(f1_samples, [2.5, 97.5]) 