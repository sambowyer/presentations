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
    May 27, 2025
  </span>
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

<<< @/snippets/snippet1.py



<!-- Notes for TL;DR -->

---

# Central Limit Theorem (CLT)

<div class="mt-8 p-4 bg-blue-100 border-l-4 border-blue-500">

If $X_1, \dots, X_N$ are <span class="highlight-red">IID</span> r.v.s with mean $\mu \in \R$ and finite variance $\sigma^2$, then 
  $$\sqrt{N} (\hat{\mu} - \mu) \xrightarrow{d} \mathcal{N} \left( 0, \sigma^2 \right) \; \text{as } \text{\colorbox{crimson}{\color{white}{$N \rightarrow \infty$}}},$$
  where $\hat{\mu} = \frac{1}{N}\sum_{i=1}^N X_i$ is the sample mean.
{v-click} 
</div>

<div class="h-10"></div>

We estimate $\sigma^2$ with 
$$\hat{\sigma}^2 = \frac{1}{N-1} \sum_{i=1}^N (X_i - \hat{\mu})^2.$$

<!-- Notes for State of evals -->


---

# Central Limit Theorem (CLT) - Confidence Intervals

We construct CLT-based confidence intervals at confidence level $1-\alpha$ as 
$$\text{CI}_{1-\alpha}(\mu) = \hat{\mu} \pm z_{\alpha/2} \text{SE}(\hat{\mu})$$
where $z_{\alpha/2}$ is the $100(1-\alpha/2)$-th percentile of the standard normal distribution and 
$$ \text{SE}(\hat{\mu}) = \sqrt{\hat{\sigma}^2 / N}$$
is the standard error of the sample mean.

<div class="mt-8 p-4 bg-blue-100 border-l-4 border-blue-500">
For binary data (e.g. correct/incorrect) we can use the Bernoulli variance formula:

$$X_i \sim \text{Bernoulli}(\theta),$$
$$\text{SE}(\hat{\theta}) = \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{N}}$$
</div>



<!-- Notes for State of evals -->

---

# Real-world failures
As models get better (and more expensive), benchmarks get harder and smaller, posing problems for the CLT.
(E.g. Math Arena's AIME II 2025 Benchmark has N=15 competition maths problems.)
- Error bars can collapse to <span class="highlight-red">zero-width.</span>
- Error bars can <span class="highlight-red">extend past \[0,1\].</span>

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

$$
\begin{aligned}
\theta &\sim \text{Beta}(1, 1) = \text{Uniform}[0, 1] \\
y_i &\sim \text{Bernoulli}(\theta) \; \text{for } i=1,\dots N \\
\mathbb{P}(\theta | {y_{1:N}}) &= \text{Beta}\left(1+\sum_{i=1}^N y_i, 1 + \sum_{i=1}^N (1-y_i)\right)
\end{aligned}
$$

<div class="h-2"></div>

Construct a quantile-based Bayesian *credible interval* for $\theta$ from the posterior.

<<< @/snippets/snippet1_bayes.py

<!-- Notes for Beta-Binomial Model -->

---

# Frequentist vs. Bayesian Intervals

- Frequentist *confidence interval:*
    - <span class="highlight-red">The parameter is fixed but unknown,</span> the interval is a random variable depending on the data.
    - "*If we repeated the experiment many times, $100 \times (1-\alpha)\%$ of the time the interval would contain the true parameter.*"
- Bayesian *credible interval:*
    - <span class="highlight-red">The parameter is random,</span> we infer the posterior distribution of the parameter given the data.
    - "*There is a $100 \times (1-\alpha)\%$ probability that the interval contains the true parameter. (Under some modelling assumptions.)*"




---

# Interval Comparison

We'll focus on two metrics for evaluating intervals:
- <span class="highlight-red">Coverage</span>
    - What proportion of the time does a $1-\alpha$ confidence-level interval *actually contain* the true underlying value of $\theta$? 
    - Ideally, this should match the _nominal_ coverage level of $1-\alpha$.
    - (This is a frequentist measure really, but still a useful one for evaluating Bayesian methods too.)
- <span class="highlight-red">Width</span> 
    - Ideally, our intervals would be as tight as possible.

<div class="h-2"></div>

---
transition: slide-left
---

# Experiment Setup

We have to rely on synthetic data so that we *know* the true parameter $\theta$.

- Draw $\theta \sim \text{Uniform}[0, 1]$.

- Draw $N \in \{10,30,100,300\}$ IID Bernoulli datapoints with parameter $\theta$.

- Construct $1-\alpha$ confidence-level intervals for $\theta$ using various methods.

- Repeat for this 200 times for each of 100 drawn values of $\theta$, for each of the 4 values of $N$.

- Compute the coverage and average width of the intervals.


---

# IID Questions Setting - Bayes vs. CLT

<img src="/img/pngs/exp4-1_bayes_clt.png" alt="IID Questions Setting" class="w-full h-full object-contain">



<!-- Notes for Comparing CLT vs. Bayes -->

---

# Alternative \#2 -- Wilson Score Intervals

$$
\text{CI}_{1-\alpha, \text{Wilson}}(\theta) = \frac{\hat{\theta} + \frac{z_{\alpha/2}^2}{2N}}{1 + \frac{z_{\alpha/2}^2}{N}} \pm \frac{\frac{z_{\alpha/2}}{2N}}{1 + \frac{z_{\alpha/2}^2}{N}}\sqrt{4N\hat{\theta}(1 - \hat{\theta}) + z_{\alpha/2}^2}
$$
- Not centered at $\hat{\theta}$
- Based on a normal approximation to the binomial distribution


<<< @/snippets/snippet1_wils.py

<!-- Notes for Wilson and Clopper-Pearson -->

---

# Alternative \#3 -- Clopper-Pearson Exact Intervals

$$
\begin{aligned}
\text{CI}_{1-\alpha, \text{CP}}(\theta) &= [\theta_\text{lower}, \theta_\text{upper}] \\
\theta_\text{lower} &= B\left(\frac{\alpha}{2}, \sum_{i=1}^N y_i, 1+\sum_{i=1}^N(1-y_i)\right) \\
\theta_\text{upper} &= B\left(1-\frac{\alpha}{2}, 1+ \sum_{i=1}^N y_i, \sum_{i=1}^N(1-y_i)\right)
\end{aligned}
$$
where $B(\alpha, a, b)$ is the $\alpha$-th quantile of the Beta$(a, b)$ distribution.
- Guaranteed to never under-cover
- Equivalent to the Bayesian posterior interval for the Beta-Binomial model without the prior on $\theta$

<<< @/snippets/snippet1_clop.py

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

## Clustered Questions
Instead of N IID questions, we have T tasks, each with K=N/T IID questions.

<!-- <div class="h-[5vh]"></div> -->

## Independent Comparisons
Compare $\theta_A$ and $\theta_B$ for two different models, with $N$ IID questions each.

<!-- <div class="h-[5vh]"></div> -->

## Paired Comparisons
Compare $\theta_A$ and $\theta_B$ for two different models, each with <u>the same</u> N IID questions.

<!-- <div class="h-[5vh]"></div> -->

## Metrics that aren't simple averages of binary results
E.g. F1 score, precision, recall, etc.


---

# Clustered Questions Setting

<div class="absolute top-4 right-4 text-sm">
(Instead of N IID questions, we have T tasks, each with K=N/T IID questions.)
</div>
<!-- (E.g. each task asks $K$ questions about a single piece of input text/data.) -->

## Generative Model
- 'Dispersion' parameter $d$ controls the range of difficulty of the questions.
- We ensure that the mean difficulty of the questions across tasks is $\theta$.

$$
\begin{aligned}
d &\sim \text{Gamma}(1, 1), \quad 
\theta \sim \text{Beta}(1, 1), \quad 
\theta_t &\sim \text{Beta}(d \theta, d (1-\theta)), \quad
y_{i,t} \sim \text{Bernoulli}(\theta_t)
\end{aligned}
$$

<div v-click></div>
<div v-click></div>

<div v-if="$slidev.nav.clicks === 1">

## Bayesian Inference
The number of successes per task is Beta-Binomial distributed: 
$$\sum_{i=1}^{N_t} y_{i,t} = Y_t \sim \text{BetaBinomial}(N_t, d \theta, d (1-\theta))$$

Get an <span class="highlight-red">importance-weighted posterior</span> for $\theta$: draw prior samples $\{(\theta^{(k)}, d^{(k)})\}_{k=1}^K$, then compute weights

$$
w^{(k)} = \prod_{t=1}^T \text{BetaBinomial}(Y_t; N_t, d^{(k)} \theta^{(k)}, d^{(k)}(1-\theta^{(k)}))
$$

Then we can compute the posterior for $\theta$ as 
$$
\theta = \frac{\sum_{k=1}^K w^{(k)} \theta^{(k)}}{\sum_{k=1}^K w^{(k)}}
$$

</div>

<div v-if="$slidev.nav.clicks === 2">

## Clustered Standard Error (CLT-based Approach)

Update the standard error to account for the clustering:

$$
\text{SE}_\text{clust.} = \sqrt{\text{SE}_\text{CLT}^2 + \frac{1}{N^2}\sum_{t=1}^T \sum_{i=1}^{N_t} \sum_{j \neq i} (y_{i,t} - \bar{y})(y_{j,t} - \bar{y})}
$$

$$\text{CI}_{1-\alpha}(\theta) = \hat{\theta} \pm z_{\alpha/2} \text{SE}_\text{clust.}$$

</div>


<!-- Notes for Clustered Questions Setting -->
---

# Clustered Questions Setting

<img src="/img/pngs/exp4-2.png" alt="Clustered Questions Setting" class="w-full h-full object-contain">


<!-- Notes for Plot 2 -->

---

# Model Comparison (Unpaired)

<!-- <div class="absolute top-4 right-4 text-sm">
(Compare $\theta_A$ and $\theta_B$ for two different models, with $N$ IID questions each.)

</div> -->

Compare $\theta_A$ and $\theta_B$ for two different models, with $N$ IID questions each.

- Compute an interval over the <span class="highlight-red">difference</span> $\theta_A - \theta_B$, check if it contains 0.
- Compute an interval over the <span class="highlight-red">odds ratio</span> $\frac{\theta_A/(1-\theta_A)}{\theta_B/(1-\theta_B)}$, check if it contains 1.


<div v-click></div>
<div v-click></div>

<div v-if="$slidev.nav.clicks === 1">

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


<div v-if="$slidev.nav.clicks === 2">

## Frequentist Approach

- Use the CLT directly for the **difference**:
$$\text{CI}_{1-\alpha}(\mu_A - \mu_B) 
    = (\hat{\mu}_A - \hat{\mu}_B)  \pm z_{\alpha/2}\,\text{SE}(\hat{\mu}_A - \hat{\mu}_B).$$

- Use an inverted Fisher's Exact Test for the **odds ratio**:
    - Much like the Clopper-Pearson exact intervals, this will never under-cover.

```python
# y_A and y_B are vectors of evals for two models
from scipy.stats.contingency import odds_ratio
S_A, S_B = y_A.sum(), y_B.sum()

result = odds_ratio([[S_A, N_A - S_A], [S_B, N_B - S_B]])
ci_or = result.confidence_interval(confidence_level=0.95, alternative='two-sided')
```   

</div>

---

# Model Comparison (Unpaired)

<!-- <div class="h-10"></div> -->

<!-- <div class="flex justify-left"> -->
<img src="/img/pngs/exp4-3.png" alt="Model Comparison (Unpaired)" class="w-full h-full object-contain max-h-80 mx-auto">
<!-- </div> -->

<div v-click>

With Bayes we can easily compute probabilities of one model being better than the other:
$$
\mathbb{P}(\theta_A > \theta_B | y_{A;1:N}, y_{B;1:N}) = \frac{1}{K} \sum_{k=1}^K {1}[\theta_A^{(k)} > \theta_B^{(k)}]
$$
where $\theta_m^{(k)} \sim p(\theta_m | y_{m, 1:N})$ are posterior samples for models $m \in \{A, B\}$.
</div>

<!-- TODO: Fill in content for Model Comparison (unpaired) -->

<!-- Notes for Model Comparison (unpaired) -->

---

# Model Comparison (Paired)

Compute intervals over the difference $\theta_A - \theta_B$, where we have access to the <span class="highlight-red">same</span> $N$ (IID) questions for both models: $\{y_{A;i}\}_{i=1}^N$ and $\{y_{B;i}\}_{i=1}^N$.


<div v-click></div>
<div v-click></div>

<div v-if="$slidev.nav.clicks === 1">

<!-- <div v-click> -->

## Frequentist Approach

Use the CLT directly for the difference:

$$\text{CI}_{1-\alpha}(\mu_A - \mu_B) 
    = (\hat{\mu}_A - \hat{\mu}_B)  \pm z_{\alpha/2}\,\text{SE}(\hat{\mu}_A - \hat{\mu}_B).$$

</div>


<div v-if="$slidev.nav.clicks === 2">
<!-- <div v-click> -->

## Bayesian Approach (Importance Sampling)

<div class="flex">
<div class="flex-1">

$$
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
This ensures we still have $y_{A;i} \sim \text{Ber}(\theta_A)$ and $y_{B;i} \sim \text{Ber}(\theta_B)$, whilst still allowing for correlation between the two models.


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

<!-- <div class="h-10"></div> -->

<!-- <div class="flex justify-left"> -->
<img src="/img/pngs/exp4-4.png" alt="Model Comparison (Paired)" class="w-full h-full object-contain mx-auto">
<!-- </div> -->

<!-- TODO: Fill in content for Model Comparison (unpaired) -->

<!-- Notes for Model Comparison (unpaired) -->


---

# Prior Mismatch

- By default, we avoid using informative/subjective priors and stick to $\theta \sim \text{Uniform}[0, 1]$.
- Bayesian methods still generally outperform CLT-based approaches when the underlying prior is different.
$$\text{e.g.} \quad \text{Beta}(100,20), \quad \mathbb{E}[\theta] = 0.83, \quad \text{Var}[\theta] = 0.0011$$

<Transform scale="0.75" origin="top">
<img src="/img/pngs/exp4-1_beta-100-20_mismatch.png" alt="Prior Mismatch" class="h-full object-contain mx-auto">
</Transform>

<!-- Notes for Prior Mismatch -->

---

# Conclusion

- Use Bayes (or Wilson), it's not hard (`scipy` or `bayes_evals`) and it's safer, and still cheap for large $N$.
- Plus you get the flexibility of Bayes! 
    - Computing probabilities $\mathbb{P}(\theta_A > \theta_B)$
    - Intervals on nonlinear functions of parameters (e.g. F1 score)
    
<img src="/img/pngs/exp4-5.png" alt="F1 score" class="w-full h-full object-contain mx-auto">

<!-- ![](/img/table_small.png) -->


<!-- Notes for Main argument -->

---
layout: section
---

# Thanks for listening!

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

## Appendix -- Clustered Importance Sampling Code

<<< snippets/snippet4.py

---

## Appendix -- Paired Importance Sampling Code

<<< @/snippets/snippet5.py{style="font-size:0.2em"}