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
transition: none
---

# Motivation

<div class="h-2"></div>

<v-clicks depth="2">

- Error bars are important for interpreting evals. 
- The CLT is the most common method for computing error bars, but it's often unwise. 
- Error bars can collapse to <span class="highlight-red">zero-width</span> or <span class="highlight-red">extend past $[0,1]$.</span>

</v-clicks>

<div class="pt-4">
  <img v-if="$slidev.nav.clicks <= 0" src="/img/langchain_plain.png" alt="Real Data Langchain Subset" class="block mx-auto max-h-80 object-contain">
  <img v-if="$slidev.nav.clicks >= 1  " src="/img/langchain_clt.png" alt="Real Data Langchain Subset" class="block mx-auto max-h-80 object-contain">
</div>


<div v-if="$slidev.nav.clicks >= 2">
<div class="absolute top-3 right-10 text-sm">
<div class="mt-0.5 p-0.5 bg-blue-100 border-l-4 border-blue-500">

CLT-based CI at confidence level $1-\alpha$ for binary data $X_i \sim \text{Bernoulli}(\theta)$:

$$
\begin{aligned}
% X_i &\sim \text{Bernoulli}(\theta) \\
\text{CI}_{1-\alpha}(\theta) &= \bar{X} \pm z_{\alpha/2} \sqrt{\frac{\bar{X}(1-\bar{X})}{N}} \\
% \bar{X} &= \frac{1}{N}\sum_{i=1}^N X_i
\end{aligned}
$$

</div>

</div>
</div>

<div class="absolute bottom-5 left-10 text-sm">
  <i>
    Langchain Typewriter Tool Use Benchmark (N=20)
  </i>
</div>


---

# Motivation

<div class="h-2"></div>

- Error bars are important for interpreting evals. 
- The CLT is the most common method for computing error bars, but it's often unwise. 
- Error bars can collapse to <span class="highlight-red">zero-width</span> or <span class="highlight-red">extend past $[0,1]$.</span>

<div class="pt-4">
  <img src="/img/langchain_clt.png" alt="Real Data Langchain Subset" class="block mx-auto max-h-80 object-contain">
</div>


<div class="absolute top-3 right-10 text-sm">
<div class="mt-0.5 p-0.5 bg-blue-100 border-l-4 border-blue-500">

CLT-based CI at confidence level $1-\alpha$ for binary data $X_i \sim \text{Bernoulli}(\theta)$:

$$
\begin{aligned}
% X_i &\sim \text{Bernoulli}(\theta) \\
\text{CI}_{1-\alpha}(\theta) &= \bar{X} \pm z_{\alpha/2} \sqrt{\frac{\bar{X}(1-\bar{X})}{N}} \\
% \bar{X} &= \frac{1}{N}\sum_{i=1}^N X_i
\end{aligned}
$$


</div>
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

<div class="h-10"></div>

<v-clicks depth="2">

- The CLT relies on a <span class="highlight-red">large $N$</span> assumption.
- As LLM capabilitites improve, constructing and running benchmarks is becoming more time-intensive.
- Researchers are increasingly using benchmarks with smaller $N$.
- We need alternative methods for computing error bars that work for both large _and_ small $N$.
</v-clicks>



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

$$p(\theta | {y_{1:N}}) = \text{Beta}\left(1+\sum_{i=1}^N y_i, 1 + \sum_{i=1}^N (1-y_i)\right)$$

</div>

<!-- <div class="h-2"></div> -->

<div v-click>

Obtain quantile-based Bayesian *credible intervals* for $\theta$ from the <span class="highlight-red">closed form posterior</span> (with confidence level $1-\alpha$).

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

<!-- ---

# Frequentist vs. Bayesian Intervals

<div class="h-10"></div>

<v-clicks depth="2">

- Frequentist *confidence interval:*
    - <span class="highlight-red">The parameter is fixed but unknown,</span> the interval is a random variable depending on the data.
    - "*If we repeated the experiment many times, $100 \times (1-\alpha)\%$ of the time the interval would contain the true parameter.*"

- Bayesian *credible interval:*
    - <span class="highlight-red">The parameter is random,</span> we infer the posterior distribution of the parameter given the data.
    - "*There is a $100 \times (1-\alpha)\%$ probability that the interval contains the true parameter. (Under some modelling assumptions.)*"

</v-clicks> -->



---

# Frequentist Alternatives

<!-- <div class="mt-2 p-0.5 bg-blue-100 border-l-4 border-blue-500">

$$
\text{CI}_{1-\alpha, \text{Wilson}}(\theta) = \frac{\hat{\theta} + \frac{z_{\alpha/2}^2}{2N}}{1 + \frac{z_{\alpha/2}^2}{N}} \pm \frac{\frac{z_{\alpha/2}}{2N}}{1 + \frac{z_{\alpha/2}^2}{N}}\sqrt{4N\hat{\theta}(1 - \hat{\theta}) + z_{\alpha/2}^2}
$$
where $z_{\alpha/2}$ is the $100(1-\alpha/2)$-th percentile of the standard normal distribution. 
</div>

<div v-click>


<div class="mt-2 p-0.5 bg-blue-100 border-l-4 border-blue-500">

$$
\text{CI}_{1-\alpha, \text{CP}}(\theta) = [\theta_\text{lower}, \theta_\text{upper}]
$$

$$
\theta_\text{lower} = B\left(\frac{\alpha}{2}, \sum_{i=1}^N y_i, 1+\sum_{i=1}^N(1-y_i)\right) \quad \text{and} \quad
\theta_\text{upper} = B\left(1-\frac{\alpha}{2}, 1+ \sum_{i=1}^N y_i, \sum_{i=1}^N(1-y_i)\right)
$$

where $B(\alpha, a, b)$ is the $\alpha$-th quantile of the Beta$(a, b)$ distribution.
</div>

</div> -->

<div class="h-10"></div>
<v-clicks depth="2">

- <span class="highlight-red">Wilson score interval</span>
  - Based on the normal approximation to the binomial distribution (but __not__ the CLT).
- <span class="highlight-red">Clopper-Pearson exact interval</span>
  - 'Worst-case' approach (very conservative method; guaranteed to never under-cover).
</v-clicks>

<div class="h-10"></div>

<div v-click> 

```python
# y is a length N binary "eval" vector
S, N = y.sum(), len(y) # total successes & questions
result = scipy.stats.binomtest(k=S, n=N)

# 95% Wilson score interval and Clopper-Pearson exact interval
wilson_ci = result.proportion_ci("wilson", 0.95)
cp_ci = result.proportion_ci("exact", 0.95)
```
</div>



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

<div v-click>

We have to rely on synthetic eval data so that we *know* the true parameter $\theta$.

</div>
<v-clicks depth="1">

- Draw $\theta \sim \text{Uniform}[0, 1]$.
- Draw $N \in \{3,10,30,100\}$ IID Bernoulli datapoints with parameter $\theta$.
- Construct intervals with various $1-\alpha$ confidence levels.
- Repeat many times and calculate the true coverage and width of the intervals.

</v-clicks>


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

Compare $\theta_A$ and $\theta_B$ for two different models, with access _only_ to $N_A, N_B, \bar{y}_A,$ and $\bar{y}_B$.

<!-- <div class="h-[5vh]"></div> -->

## Paired Comparisons

Compare $\theta_A$ and $\theta_B$ for two different models, each with <u>the same</u> $N$ IID questions and access to question-level successes $\{y_{A;i}\}_{i=1}^N$ and $\{y_{B;i}\}_{i=1}^N$.

<!-- <div class="h-[5vh]"></div> -->

## Metrics that aren't simple averages of binary results (e.g. F1 score).

</v-clicks>

---

# Conclusion

Use Bayes or Wilson Score intervals.

<v-clicks depth="2">

- It's not hard (use `scipy` or `bayes_evals`).
- It's safer than CLT-based methods.
- It's still cheap for large $N$.

</v-clicks>

<div v-click>

<div class="grid grid-cols-2 gap-4">
<div class="flex flex-col items-center">
<h2><span class="highlight-red">Paper</span></h2>
<a href="https://arxiv.org/pdf/2503.01747" target="_blank">https://arxiv.org/pdf/2503.01747</a>
<div class="h-[3vh]"></div>
<img src="/img/arxiv_qr.png" class="w-36 h-36 object-contain">
</div>

<div class="flex flex-col items-center">
<h2><span class="highlight-red">bayes_evals</span> package</h2>
<a href="https://github.com/sambowyer/bayes_evals" target="_blank">https://github.com/sambowyer/bayes_evals</a>
<div class="h-[3vh]"></div>
<img src="/img/github_qr.png" class="w-36 h-36 object-contain">
</div>
</div>
</div>