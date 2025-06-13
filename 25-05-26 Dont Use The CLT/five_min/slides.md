---
theme: apple-basic
title: "Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints"
info: |
  Presentation on the ICML 2025 Spotlight Position Paper: "Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints"
transition: slide-left
mdc: true
layout: intro
shiki:
  theme: 'catppuccin-macchiato'

---

<style>
  .highlight-red {
    background-color: crimson;
    color: white;
    padding: 2px 4px;
    border-radius: 4px;
  }
</style>

# Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints

Sam Bowyer, Laurence Aitchison, and Desi R. Ivanova

<div class="absolute bottom-20">
  <div class="flex gap-2">
    <img src="/img/sam.png" class="rounded-full w-24 h-24" alt="Sam Bowyer">
    <img src="/img/laurence.png" class="rounded-full w-24 h-24" alt="Laurence Aitchison">
    <img src="/img/desi.jpg" class="rounded-full w-24 h-24" alt="Desi R. Ivanova">
  </div>
</div>


<div class="absolute bottom-10">
  <span class="font-700">
    ICML 2025 Spotlight Position Paper
  </span>
</div>


---

# Motivation

<v-clicks depth="2">

- Error bars are important for interpreting evals.
- The CLT is the most common method for computing error bars, but it's often unwise.
- Error bars can collapse to <span class="highlight-red">zero-width</span> or <span class="highlight-red">extend past $[0,1]$.</span>

</v-clicks>

<div class="pt-4">
  <img v-if="$slidev.nav.clicks === 0" src="/img/langchain_plain.png" alt="Real Data Langchain Subset" class="block mx-auto max-h-80 object-contain">
  <img v-if="$slidev.nav.clicks >= 1" src="/img/langchain_clt.png" alt="Real Data Langchain Subset" class="block mx-auto max-h-80 object-contain">
</div>

<div class="absolute bottom-5 left-10 text-sm">
  <i>
    Langchain Typewriter Tool Use Benchmark (N=20)
  </i>
</div>

---

# Central Limit Theorem (CLT)

<div class="mt-6 p-2 bg-blue-100 border-l-4 border-blue-500">

If $X_1, \dots, X_N$ are <span class="highlight-red">IID</span> r.v.s with mean $\mu \in \R$ and finite variance $\sigma^2$, then 
  $$\sqrt{N} (\hat{\mu} - \mu) \xrightarrow{d} \mathcal{N} \left( 0, \sigma^2 \right) \; \text{as } \text{\colorbox{crimson}{\color{white}{$N \rightarrow \infty$}}},$$
  where $\hat{\mu} = \frac{1}{N}\sum_{i=1}^N X_i$ is the sample mean.


</div>
<div v-click>
<div class="mt-6 p-2 bg-blue-100 border-l-4 border-blue-500">

For binary data (e.g. correct/incorrect), $X_i \sim \text{Bernoulli}(\theta)$, we construct the CLT-based confidence interval at confidence level $1-\alpha \in [0,1]$ as

$$\text{CI}_{1-\alpha}(\theta) = \hat{\theta} \pm z_{\alpha/2} \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{N}},$$

where $z_{\alpha/2}$ is the $100(1-\alpha/2)$-th percentile of $\mathcal{N}(0,1)$ and $\hat{\theta} = \frac{1}{N}\sum_{i=1}^N X_i$ is the sample mean.
</div>
</div>

---

# Alternative \#1 -- Beta-Binomial Model
Treat the data as IID Bernoulli with a uniform prior on the parameter $\theta$.

$${1-2|all}
\begin{aligned}
\theta &\sim \text{Beta}(1, 1) = \text{Uniform}[0, 1] \\
y_i &\sim \text{Bernoulli}(\theta) \; \text{for } i=1,\dots N 
\end{aligned}
$$

<div v-click>

We say $y_i$ is correct if $y_i = 1$ and incorrect if $y_i = 0$. (Think of $\theta$ as the probability of correctness.)

</div>

<div v-click>

$$\mathbb{P}(\theta | {y_{1:N}}) = \text{Beta}\left(1+\sum_{i=1}^N y_i, 1 + \sum_{i=1}^N (1-y_i)\right)$$

</div>

<div class="h-2"></div>

<div v-click>

Construct a quantile-based Bayesian *credible interval* for $\theta$ from the <span class="highlight-red">closed form posterior</span>.

</div>

<div v-click>

<!-- <<< @/snippets/snippet1_bayes.py -->
```python
# y is a length N binary "eval" vector
S, N = y.sum(), len(y) # total successes & questions

# Bayesian Credible interval
posterior = scipy.stats.beta(1+S, 1+(N-S))
bayes_ci = posterior.interval(confidence=0.95)
```

</div>

<!-- Notes for Beta-Binomial Model -->

---

# Frequentist vs. Bayesian Intervals

<div class="h-10"></div>

<v-clicks depth="2">

- Frequentist *confidence interval:*
    - <span class="highlight-red">The parameter is fixed but unknown,</span> the interval is a random variable depending on the data.
    - "*If we repeated the experiment many times, $100 \times (1-\alpha)\%$ of the time the interval would contain the true parameter.*"

- Bayesian *credible interval:*
    - <span class="highlight-red">The parameter is random,</span> we infer the posterior distribution of the parameter given the data.
    - "*There is a $100 \times (1-\alpha)\%$ probability that the interval contains the true parameter. (Under some modelling assumptions.)*"

</v-clicks>



---

# Interval Comparison

We'll focus on two metrics for evaluating intervals:

<v-clicks depth="2">

- <span class="highlight-red">Coverage</span>
    - What proportion of the time does a $1-\alpha$ confidence-level interval *actually contain* the true underlying value of $\theta$? 
    - Ideally: _actual_ coverage $=$ _nominal_ coverage (i.e. $1-\alpha$).
- <span class="highlight-red">Width</span> 
    - Ideally, our intervals would be as tight as possible.

</v-clicks>

<div class="h-2"></div>

---
transition: slide-left
---

# Experiment Setup

<div class="h-15"></div>

We have to rely on synthetic data so that we *know* the true parameter $\theta$.

<v-clicks depth="1">

- Draw $\theta \sim \text{Uniform}[0, 1]$.

- Draw $N \in \{3,10,30,100\}$ IID Bernoulli datapoints with parameter $\theta$.

- Construct $1-\alpha$ confidence-level intervals for $\theta$ using both methods with various $\alpha$ values.

- Repeat for this 20,000 times for each of the 4 values of $N$.

- Compute the coverage and average width of the intervals.

</v-clicks>


---

# IID Questions Setting - Bayes vs. CLT

<img src="/img/pngs/exp4-1_bayes_clt.png" alt="IID Questions Setting" class="w-full h-full object-contain">



<!-- Notes for Comparing CLT vs. Bayes -->

---

# Alternative \#2 -- Wilson Score Intervals

<div class="mt-8 p-4 bg-blue-100 border-l-4 border-blue-500">

$$
\text{CI}_{1-\alpha, \text{Wilson}}(\theta) = \frac{\hat{\theta} + \frac{z_{\alpha/2}^2}{2N}}{1 + \frac{z_{\alpha/2}^2}{N}} \pm \frac{\frac{z_{\alpha/2}}{2N}}{1 + \frac{z_{\alpha/2}^2}{N}}\sqrt{4N\hat{\theta}(1 - \hat{\theta}) + z_{\alpha/2}^2}
$$
where $z_{\alpha/2}$ is the $100(1-\alpha/2)$-th percentile of the standard normal distribution. 
</div>

<v-clicks depth="2">

- Not centered at $\hat{\theta}$.

- Based on a normal approximation to the binomial distribution (but __not__ the CLT).
</v-clicks>

<div v-click>

```python
# y is a length N binary "eval" vector
S, N = y.sum(), len(y) # total successes & questions
result = scipy.stats.binomtest(k=S, n=N)

# 95% Wilson score interval
wilson_ci = result.proportion_ci("wilson", 0.95)
```
<!-- <<< @/snippets/snippet1_wils.py -->

</div>

<!-- Notes for Wilson and Clopper-Pearson -->

---

# Alternative \#3 -- Clopper-Pearson Exact Intervals

<div class="mt-3 p-0.5 bg-blue-100 border-l-4 border-blue-500">
<!-- $$
\begin{aligned}
\text{CI}_{1-\alpha, \text{CP}}(\theta) &= [\theta_\text{lower}, \theta_\text{upper}] \\
\theta_\text{lower} &= B\left(\frac{\alpha}{2}, \sum_{i=1}^N y_i, 1+\sum_{i=1}^N(1-y_i)\right) \\
\theta_\text{upper} &= B\left(1-\frac{\alpha}{2}, 1+ \sum_{i=1}^N y_i, \sum_{i=1}^N(1-y_i)\right)
\end{aligned}
$$ -->

$$
\text{CI}_{1-\alpha, \text{CP}}(\theta) = [\theta_\text{lower}, \theta_\text{upper}]
$$

$$
\theta_\text{lower} = B\left(\frac{\alpha}{2}, \sum_{i=1}^N y_i, 1+\sum_{i=1}^N(1-y_i)\right) \quad \text{and} \quad
\theta_\text{upper} = B\left(1-\frac{\alpha}{2}, 1+ \sum_{i=1}^N y_i, \sum_{i=1}^N(1-y_i)\right)
$$

where $B(\alpha, a, b)$ is the $\alpha$-th quantile of the Beta$(a, b)$ distribution.
</div>
<v-clicks depth="2">

- Guaranteed to never under-cover (very conservative method; 'worst-case' approach).
  - Contains all $\theta \in [0,1]$ that would not reject $H_0: \theta = \hat{\theta}$ in favour of $H_1: \theta \neq \hat{\theta}$ at confidence level $\alpha$.
- Equivalent to the Bayesian interval with the uniform prior on $\theta$ removed.
</v-clicks>

<div v-click>
<!-- <<< @/snippets/snippet1_clop.py -->
```python
# y is a length N binary "eval" vector
S, N = y.sum(), len(y) # total successes & questions
result = scipy.stats.binomtest(k=S, n=N)
# 95% Clopper-Pearson exact interval
cp_ci = result.proportion_ci("exact", 0.95)
```

</div>



<!-- Notes for Wilson and Clopper-Pearson -->

---

# IID Questions Setting

<img src="/img/pngs/exp4-1.png" alt="IID Questions Setting" class="w-full h-full object-contain">


<!-- Notes for Plot 2 -->
<!-- - (Briefly mention bootstrap -- it's what OpenAI suggest (ref?) bc it's flexible but it's bad)
- Basically: Bayes and Wilson are the best (they make binary assumptions about data and aren't overly cautious, like CP) -->


---

# Other Eval Settings

<div class="h-[1vh]"></div>

<v-clicks depth="2">

## Clustered Questions

Instead of $N$ IID questions, we have $T$ tasks, each with $N_t$ IID questions.

<!-- <div class="h-[5vh]"></div> -->

## Independent Comparisons

Compare $\theta_A$ and $\theta_B$ for two different models, with access _only_ to $N_A, N_B, \hat{\theta}_A,$ and $\hat{\theta}_B$.

<!-- <div class="h-[5vh]"></div> -->

## Paired Comparisons

Compare $\theta_A$ and $\theta_B$ for two different models, each with <u>the same</u> $N$ IID questions and access to question-level successes $\{y_{A;i}\}_{i=1}^N$ and $\{y_{B;i}\}_{i=1}^N$.

<!-- <div class="h-[5vh]"></div> -->

## Metrics that aren't simple averages of binary results (e.g. F1 score).

</v-clicks>

---

# Conclusion

<v-clicks depth="2">

- Use Bayes, it's not hard (`scipy` or `bayes_evals`), it's safer, and it's still cheap for large $N$.
    - Computing probabilities $\mathbb{P}(\theta_A > \theta_B)$ is easy.
    - Better performance than CLT-based methods.
    - Robust to mismatched priors --- many ablations provided in the paper's appendix.

</v-clicks>
<div v-if="$slidev.nav.clicks === 4">

<img src="/img/pngs/exp4-1_beta-100-20_mismatch.png" alt="Mismatched prior plot" class="w-70% object-contain object-top mx-auto">
<div class="absolute bottom-5 right-10 text-sm">

<!-- Instead of $\theta \sim \text{Uniform}[0, 1]$, we have -->
$$
\begin{aligned} 
\theta &\sim \text{Beta}(100,20) \\
\mathbb{E}[\theta] &= 0.83 \\
\text{Var}[\theta] &= 0.034^2
\end{aligned}
$$
</div>
</div>

<!-- ![](/img/table_small.png) -->


<!-- Notes for Main argument -->

---
layout: section
---

# Thanks!

<div class="h-[8vh]"></div>

<div class="grid grid-cols-2 gap-4">
<div class="flex flex-col items-center">
<h2><span class="highlight-red">Paper</span></h2>
<a href="https://arxiv.org/pdf/2503.01747" target="_blank">https://arxiv.org/pdf/2503.01747</a>
<div class="h-[3vh]"></div>
<img src="/img/arxiv_qr.png" class="w-48 h-48 object-contain">
</div>

<div class="flex flex-col items-center">
<h2><span class="highlight-red">bayes_evals</span> package</h2>
<a href="https://github.com/sambowyer/bayes_evals" target="_blank">https://github.com/sambowyer/bayes_evals</a>
<div class="h-[3vh]"></div>
<img src="/img/github_qr.png" class="w-48 h-48 object-contain">
</div>
</div>
