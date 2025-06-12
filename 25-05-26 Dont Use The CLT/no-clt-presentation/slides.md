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

<!-- ![](/img/title.png) -->

<!-- <div class="h-2"></div> -->

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
    June 12, 2025
  </span>
</div>

---

<!-- <div class="h-0"></div> -->

<!-- <div class="grid grid-cols-2 gap-4">
  <div>
    <h2>Recommendation</h2>
    <p>Use Bayes or Wilson Score Intervals, not the CLT.</p>
  </div>
  <div>
    <img src="/img/anthropic_blog.png" alt="Anthropic Blog Post" class="w-full">
  </div>
</div> -->

<!-- <div class="h-[20vh]"></div> -->
<div class="flex justify-center">
<img src="/img/anthropic_blog2.png" alt="Anthropic Blog Post" class="w-5/6 h-full object-contain">
</div>

---
transition: none
---

# TL;DR

- Error bars are important for evals{v-click}
- CLT-based methods are (increasingly) unwise{v-click}

<div class="pt-4">
  <img v-if="$slidev.nav.clicks === 1" src="/img/langchain_plain.png" alt="Real Data Langchain Subset" class="block mx-auto max-h-80 object-contain">
  <img v-if="$slidev.nav.clicks === 2" src="/img/langchain_clt.png" alt="Real Data Langchain Subset" class="block mx-auto max-h-80 object-contain">
</div>

<!-- {t-click} -->

<!-- <<< @/snippets/snippet1.py -->



<!-- Notes for TL;DR -->

---
transition: slide-left
---

# TL;DR

- Error bars are important for evals
- CLT-based methods are (increasingly) unwise
- We can do a lot better, very easily

<!-- {n-click} -->

<!-- <<< @/snippets/snippet1.py -->

```python
# y is a length N binary "eval" vector
from scipy.stats import binomtest, beta

S, N = y.sum(), len(y) # total successes & questions
result = binomtest(k=S, n=N)

# 95% Wilson score and Clopper-Pearson intervals
wilson_ci = result.proportion_ci("wilson", 0.95)
cp_ci = result.proportion_ci("exact", 0.95)

# Bayesian Credible interval
posterior = beta(1+S, 1+(N-S))
bayes_ci = posterior.interval(confidence=0.95)
```



<!-- Notes for TL;DR -->

---

# Central Limit Theorem (CLT)

<div class="h-15"></div>

<div class="mt-8 p-4 bg-blue-100 border-l-4 border-blue-500">

If $X_1, \dots, X_N$ are <span class="highlight-red">IID</span> r.v.s with mean $\mu \in \R$ and finite variance $\sigma^2$, then 
  $$\sqrt{N} (\hat{\mu} - \mu) \xrightarrow{d} \mathcal{N} \left( 0, \sigma^2 \right) \; \text{as } \text{\colorbox{crimson}{\color{white}{$N \rightarrow \infty$}}},$$
  where $\hat{\mu} = \frac{1}{N}\sum_{i=1}^N X_i$ is the sample mean.


</div>

<div class="h-25"></div>

<div v-click>

(We generally estimate $\sigma^2 \approx \hat{\sigma}^2 = \frac{1}{N-1} \sum_{i=1}^N (X_i - \hat{\mu})^2.$)
</div>

<!-- Notes for State of evals -->


---

# Central Limit Theorem (CLT) - Confidence Intervals

<div class="mt-8 p-4 bg-blue-100 border-l-4 border-blue-500">

We construct CLT-based confidence intervals at confidence level $1-\alpha \in [0,1]$ as 
$$\text{CI}_{1-\alpha}(\mu) = \hat{\mu} \pm z_{\alpha/2} \text{SE}(\hat{\mu}),$$
where $z_{\alpha/2}$ is the $100(1-\alpha/2)$-th percentile of the standard normal distribution and 
$$ \text{SE}(\hat{\mu}) = \sqrt{\hat{\sigma}^2 / N}$$
is the standard error of the sample mean.
</div>

<div v-click>

For binary data (e.g. correct/incorrect), $X_i \sim \text{Bernoulli}(\theta)$, we can use the Bernoulli variance formula:

$$\text{SE}(\hat{\theta}) = \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{N}}.$$
</div>



<!-- Notes for State of evals -->

---

# Real-world failures
As models get better (and more expensive), benchmarks get harder and smaller, posing problems for the CLT.
(E.g. Math Arena's AIME II 2025 Benchmark has N=15 competition maths problems.)

<div v-click>
- Error bars can collapse to <span class="highlight-red">zero-width.</span>
</div>
<div v-click>
- Error bars can <span class="highlight-red">extend past [0,1].</span>
</div>


<div class="grid grid-cols-2 gap-4">
  <div>
    <img src="/img/langchain_clt.png" alt="Langchain CLT Example" class="w-full h-full object-contain">
    <span class="text-sm">
      <i>
        Langchain Typewriter Tool Use Benchmark (N=20)
      </i>
    </span>
  </div>
  <div>
    <img src="/img/pngs/real_data_matharena_aime_II_full.png" alt="Math Arena Example" class="w-full h-full object-contain">
    <span class="text-sm">
      <i>
        Math Arena's AIME II 2025 Benchmark (N=15)
      </i>
    </span>
  </div>
</div>


<!-- Notes for Real-world failures -->


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
    - (This is a frequentist measure really, but still a useful one for evaluating Bayesian methods too.)
- <span class="highlight-red">Width</span> 
    - Ideally, our intervals would be as tight as possible.
    - _"For a certain width of interval, method X achieves the highest coverage."_

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

<div class="h-[10vh]"></div>

<div class="grid grid-cols-2 gap-4">
  <div>
    <h2>Recommendation</h2>
    <p>Use Bayes or Wilson Score Intervals, not the CLT.</p>
  </div>
  <div>
    <img src="/img/anthropic_blog.png" alt="Anthropic Blog Post" class="w-full">
  </div>
</div>

<div class="h-[20vh]"></div>


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

# Clustered Questions Setting

<div class="absolute top-4 right-4 text-xs">

(Instead of $N$ IID questions, we have $T$ tasks, each with $N_t$ IID questions.)
</div>
<!-- (E.g. each task asks $K$ questions about a single piece of input text/data.) -->
<v-clicks depth="1">

Some tasks are easier than others. (E.g. each task asks $K$ questions about a single piece of input text.)
- 'Dispersion' parameter $d$ controls the range of difficulty across tasks.
$$d \sim \text{Gamma}(1, 1)$$
- $\theta$ is the mean difficulty of the questions across tasks.
$$
\theta \sim \text{Beta}(1, 1) = \text{Uniform}[0, 1]
$$
- $\theta_t$ is the difficulty of the questions in task $t$ (we ensure that $\mathbb{E}[\theta_t] = \theta$).
$$
\theta_t \sim \text{Beta}(d \theta, d (1-\theta))
$$
- If $d$ is small, then $\theta_t$ is close to $\theta$ for all tasks.
- If $d$ is large, then $\theta_t$ is more variable across tasks.
</v-clicks>





---

# Clustered Questions Setting

<div class="absolute top-4 right-4 text-xs">

(Instead of $N$ IID questions, we have $T$ tasks, each with $N_t$ IID questions.)
</div>
<!-- (E.g. each task asks $K$ questions about a single piece of input text/data.) -->


## Generative Model
<v-clicks depth="1">

- 'Dispersion' parameter $d$ controls the range of difficulty of the questions, whilst maintaining $\mathbb{E}[\theta_t] = \theta$.
</v-clicks>

<div v-click>

$$
\begin{aligned}
d &\sim \text{Gamma}(1, 1), \quad 
\theta \sim \text{Beta}(1, 1), \quad 
\theta_t &\sim \text{Beta}(d \theta, d (1-\theta)), \quad
y_{i,t} \sim \text{Bernoulli}(\theta_t)
\end{aligned}
$$
</div>

<div v-click></div>
<div v-click></div>
<div v-click></div>
<div v-click></div>

<div v-if="$slidev.nav.clicks >= 3 && $slidev.nav.clicks <= 5">

## Bayesian Inference
- Can we integrate out the per-task difficulty parameters $\theta_t$?

<div v-if="$slidev.nav.clicks >= 4">

- Yes! The number of successes per task is Beta-Binomial distributed: 
$$\sum_{i=1}^{N_t} y_{i,t} = Y_t \sim \text{BetaBinomial}(N_t, d \theta, d (1-\theta))$$

<div v-if="$slidev.nav.clicks >= 5">

Get an <span class="highlight-red">importance-weighted posterior</span> for $\theta$: draw prior samples $\{(\theta^{(k)}, d^{(k)})\}_{k=1}^K$, then compute weights

$$
w^{(k)} = \prod_{t=1}^T p(Y_t | \theta^{(k)}, d^{(k)})
$$

</div>
</div>
</div>

<div v-if="$slidev.nav.clicks === 6">

## Clustered Standard Error (CLT-based Approach)

Update the standard error to account for the clustering:

<div class="bg-blue-100 border-l-4 border-blue-500 p-4">
$$
\text{SE}_\text{clust.} = \sqrt{\text{SE}_\text{CLT}^2 + \frac{1}{N^2}\sum_{t=1}^T \sum_{i=1}^{N_t} \sum_{j \neq i} (y_{i,t} - \bar{y})(y_{j,t} - \bar{y})}
$$

$$\text{CI}_{1-\alpha}(\theta) = \hat{\theta} \pm z_{\alpha/2} \text{SE}_\text{clust.}$$
</div>


</div>


<!-- Notes for Clustered Questions Setting -->
---

# Clustered Questions Setting
<div class="absolute top-4 right-4 text-xs">

(Dotted lines are IID-assuming methods, solid lines are clustered methods.)
</div>

<img src="/img/pngs/exp4-2.png" alt="Clustered Questions Setting" class="w-full h-full object-contain">


<!-- Notes for Plot 2 -->

---

# Model Comparison (Unpaired)

<!-- <div class="absolute top-4 right-4 text-sm">
(Compare $\theta_A$ and $\theta_B$ for two different models, with $N$ IID questions each.)

</div> -->

Compare $\theta_A$ and $\theta_B$ for two different models, with access _only_ to $N = N_A = N_B, \hat{\theta}_A,$ and $\hat{\theta}_B$.

<v-clicks depth="2">

- Compute an interval over the <span class="highlight-red">difference</span> $\theta_A - \theta_B$ check if it's positive.
- Compute an interval over the <span class="highlight-red">odds ratio</span> $\frac{\theta_A/(1-\theta_A)}{\theta_B/(1-\theta_B)}$, check if it's greater than 1.
</v-clicks>

<div v-click></div>
<div v-click></div>

<div v-if="$slidev.nav.clicks === 3">

## Bayesian Approach
Obtain a posterior for model A and a posterior for model B, using the earlier Beta-Binomial model.

<!-- <<< @/snippets/snippet2.py -->
```python
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
```

</div>


<div v-if="$slidev.nav.clicks === 4 || $slidev.nav.clicks === 5">

## Frequentist Approach

- Use the CLT for the **difference** and add $A$ and $B$'s squared standard errors:
$$
\text{CI}_{1-\alpha}(\theta_A - \theta_B) 
    % = (\hat{\theta}_A - \hat{\theta}_B)  \pm z_{\alpha/2}\,\sqrt{\text{SE}(\hat{\theta}_A)^2 + \text{SE}(\hat{\theta}_B)^2}.
    = (\hat{\theta}_A - \hat{\theta}_B)  \pm z_{\alpha/2}\,\sqrt{S_A^2/N_A + S_B^2/N_B}.
$$

<div v-click>

</div>

<div v-if="$slidev.nav.clicks === 5">

- Use an inverted Fisher's Exact Test for the **odds ratio** (this will never under-cover):
    - Equivalent to Bayesian intervals with strong priors: $\theta_A \sim \text{Beta}(1, 0)$ and $\theta_B \sim \text{Beta}(0, 1)$.

```python
# y_A and y_B are vectors of evals for two models
from scipy.stats.contingency import odds_ratio
S_A, S_B = y_A.sum(), y_B.sum()
result = odds_ratio([[S_A, N_A - S_A], [S_B, N_B - S_B]])
ci_or = result.confidence_interval(confidence_level=0.95, alternative='two-sided')
```
</div>
</div>

---

# Model Comparison (Unpaired)

<!-- <div class="h-10"></div> -->

<!-- <div class="flex justify-left"> -->
<img src="/img/pngs/exp4-3.png" alt="Model Comparison (Unpaired)" class="w-9/10 h-full object-contain max-h-80 mx-auto">
<!-- </div> -->

<div v-click>
<div class="mt-8 p-0.5 bg-blue-100 border-l-4 border-blue-500">

<span class="highlight-red">Bayesian Bonus:</span> we can easily compute probabilities of one model being better than the other:
$$
\mathbb{P}(\theta_A > \theta_B | y_{A;1:N}, y_{B;1:N}) = \frac{1}{K} \sum_{k=1}^K {1}[\theta_A^{(k)} > \theta_B^{(k)}].
$$
where $\theta_m^{(k)} \sim p(\theta_m | y_{m, 1:N})$ are posterior samples for models $m \in \{A, B\}$. 
</div>

</div>

<!-- TODO: Fill in content for Model Comparison (unpaired) -->

<!-- Notes for Model Comparison (unpaired) -->

---

# Model Comparison (Paired)

Compute intervals over the difference $\theta_A - \theta_B$, where we have access to the <span class="highlight-red">same</span> $N$ (IID) questions for both models: $\{y_{A;i}\}_{i=1}^N$ and $\{y_{B;i}\}_{i=1}^N$.


<div v-click></div>


<div v-if="$slidev.nav.clicks === 1">

<!-- <div v-click> -->

## Frequentist Approach

Use the CLT directly for the difference $D_i = y_{A;i} - y_{B;i}$:

$$
\begin{aligned}
\hat{\theta}_D &= \frac{1}{N}\sum_{i=1}^N D_i, \\
\text{CI}_{1-\alpha}(\theta_A - \theta_B) 
    &= \hat{\theta}_D  \pm z_{\alpha/2}\,\text{SE}(\hat{\theta}_D).
\end{aligned}
$$

</div>
<div v-click></div>

<div v-if="$slidev.nav.clicks >= 2">
<!-- <div v-click> -->

## Bayesian Approach (Importance Sampling)

<!-- <div v-click></div>
<div v-click></div>
<div v-click></div> -->

<div class="flex">
<div class="flex-1">

$${1|2,3|1-4|5-6|all}
\begin{aligned}
\theta_A, \theta_B &\sim \text{Beta}(1, 1) = \text{Uniform}[0, 1], \\
\hat{\rho} &\sim \text{Beta}(4, 2), \\
\rho &= 2\hat{\rho} - 1, \\
(a_i,b_i) &\sim \mathcal{N}\left(\begin{pmatrix}\Phi^{-1}(\theta_A) \\ \Phi^{-1}(\theta_B)\end{pmatrix}, \begin{pmatrix}1 & \rho \\ \rho & 1\end{pmatrix}\right), \\
y_{A;i} &= 1[a_i > 0],\\
y_{B;i} &= 1[b_i > 0].
\end{aligned}
$$

where $\Phi$ is the standard normal CDF.

<div v-click>

Ensures $y_{A;i} \sim \text{Ber}(\theta_A)$ and $y_{B;i} \sim \text{Ber}(\theta_B)$, whilst still allowing for correlation between the two models.

</div>


</div>

<div class="flex-1">
<img src="/img/paired_joint.png" class="object-contain max-h-80">
</div>
</div>



</div>




<!-- TODO: Fill in content for Model Comparison (paired) -->

<!-- Notes for Model Comparison (paired) -->


---

# Model Comparison (Paired)

<div class="absolute top-4 right-4 text-xs">
(Dotted lines are unpaired methods, solid lines are paired methods.)
</div>

<!-- <div class="h-10"></div> -->

<!-- <div class="flex justify-left"> -->
<img src="/img/pngs/exp4-4.png" alt="Model Comparison (Paired)" class="w-full h-full object-contain mx-auto">
<!-- </div> -->

<!-- TODO: Fill in content for Model Comparison (unpaired) -->

<!-- Notes for Model Comparison (unpaired) -->


---

# Prior Mismatch

<v-clicks depth="2">

- By default, we avoid using informative/subjective priors and stick to $\theta \sim \text{Uniform}[0, 1]$.
- Bayesian methods still generally outperform CLT-based approaches when the underlying prior is different.

</v-clicks>

<div v-click>

$$\text{e.g.} \quad \text{Beta}(100,20), \quad \mathbb{E}[\theta] = 0.83, \quad \text{Var}[\theta] = 0.034^2$$

<Transform scale="0.75" origin="top">
<img src="/img/pngs/exp4-1_beta-100-20_mismatch.png" alt="Prior Mismatch" class="h-full object-contain mx-auto">
</Transform>

</div>
<!-- Notes for Prior Mismatch -->

---

# Conclusion

<v-clicks depth="2">

- Use Bayes (or Wilson), it's not hard (`scipy` or `bayes_evals`), it's safer, and it's still cheap for large $N$.
- Plus you get the flexibility of Bayes! 
    - Computing probabilities $\mathbb{P}(\theta_A > \theta_B)$.
    - Intervals on nonlinear functions of sample means e.g. F1 score (harmonic mean of precision and recall).

</v-clicks>

<div v-if="$slidev.nav.clicks === 4">

<img src="/img/pngs/exp4-5.png" alt="F1 score" class="w-full h-full object-contain mx-auto">
</div>

<!-- ![](/img/table_small.png) -->


<!-- Notes for Main argument -->

---
layout: section
---

# Questions?

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

---

---

# Summary Table

<div class="h-[3vh]"></div>

![](/img/table.png)

---

## Appendix -- Clustered Importance Sampling Code

<!-- <<< snippets/snippet4.py -->

```python
S_t, N_t: np.arrays of length T with total successes & questions per task
# set number of samples, K
K = 10_000

# get K samples from the prior (with extra dimension for broadcasting over tasks)
thetas = np.random.beta(1,1, size=(K,1))
ds = np.random.gamma(1,1, size=(K,1))

# obtain weights via the likelihood (sum the per-task log-probs)
log_weights = scipy.stats.betabinom(N_t, (ds*thetas), (ds*(1-thetas))).logpmf(S_t).sum(-1)

# normalise the weights
weights = np.exp(log_weights - log_weights.max())
weights /= weights.sum()

# obtain samples from the posterior
posterior = thetas[np.random.choice(K, size=K, replace=True, p=weights)]

# Bayesian credible interval
bayes_ci = np.percentile(posterior, [2.5, 97.5])
```

---

## Appendix -- Paired Importance Sampling Code

<!-- <<< @/snippets/snippet5.py{style="font-size:0.2em"} -->

```python {style="font-size:0.2em"}
# y_A, y_B: length N binary "eval" vectors
from binorm import binorm_cdf # 2D Gaussian CDF, defined elsewhere
K = 10_000
# get K samples from the prior
theta_As, theta_Bs, rhos = np.random.beta(1,1, size=K), np.random.beta(1,1,size=K), 2*np.random.beta(4,2, size=K) - 1
# 2x2 contingency table (flattened)
S = (y_A * y_B).sum(-1)             # S = A correct,   B correct
T = (y_A * (1 - y_B)).sum(-1)       # T = A correct,   B incorrect
U = ((1 - y_A) * y_B).sum(-1)       # U = A incorrect, B correct
V = ((1 - y_A) * (1 - y_B)).sum(-1) # V = A incorrect, B incorrect
# calculate the bivariate normal mean
mu_As, mu_Bs = scipy.stats.norm(0,1).ppf(theta_As), scipy.stats.norm(0,1).ppf(theta_Bs)
# Calculate probabilities of each cell in the 2x2 table
theta_V = binorm_cdf(x1=0, x2=0, mu1=mu_As, mu2=mu_Bs, sigma1=1, sigma2=1, rho=rhos)
theta_S = theta_As + theta_Bs + theta_V - 1
theta_T = 1 - theta_Bs - theta_V
theta_U = 1 - theta_As - theta_V
# (probabilities may be very small and negative instead of 0)
valid_idx = (theta_S > 0) & (theta_T > 0) & (theta_U > 0) & (theta_V > 0) 
log_weights = S*np.log(theta_S[valid_idx]) + T*np.log(theta_T[valid_idx]) + \
              U*np.log(theta_U[valid_idx]) + V*np.log(theta_V[valid_idx])
# normalise the weights and obtain samples from the posterior
weights = np.zeros(K)
weights[valid_idx] = np.exp(log_weights - log_weights.max())
posterior = (theta_As - theta_Bs)[np.random.choice(K, size=K, replace=True, p=weights/weights.sum())]
bayes_ci = np.percentile(posterior, [2.5, 97.5])
```