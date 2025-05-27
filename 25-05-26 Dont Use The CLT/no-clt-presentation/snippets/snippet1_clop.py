# y is a length N binary "eval" vector
S, N = y.sum(), len(y) # total successes & questions
result = scipy.stats.binomtest(k=S, n=N)
# 95% Clopper-Pearson exact interval
cp_ci = result.proportion_ci("exact", 0.95)