we require stationarity and diminishing adaptation AND containment:

stationarity: every $\theta \in \Theta$ has $\pi$-ergodicity

diminishing adaptation: want less and less adaptation as time goes on; either
	1. by a decreasing step-size (or equivalent), either to zero exactly or in limit (this is what Andrieu & Thom consider)
	2. by decreasing the "amount" by which we are adapting $\theta$ (e.g. Haario et al. using $N(0, \lambda \Sigma)$ with $\Sigma$ an empirical estimate of the covariance of $\Sigma$ and $\lambda = 2.38^2/d$ or something w/ $d = dim(X)$.)

containment: time from $X_n$ to $\pi$ with adaptoive transitions remain bounded in prob. as $n \to \infty$
	- (I think that Andrieu & Thom ignore this b/c they only consider the case where diminishing adaptation is achieved by a step size that is zero forall steps $i > k_i$. ("Vanishing adaptation")
	In this case we end up just using a single $\theta$ value and thus our stationarity assumption means we also have containment.)
	(N.B. In general, there are lots of different ways to ensure/prove containment for different adaptive MCMC strategies)
	- talked about more in the Warwick notes: best to have common drift functions: 
		- want convergence to $\pi$ to happen in similar ways for different groups of initial/early params $\theta$
			- is this similar to the reasoning for "adaptive truncation" by Andrieu et al. that helps ensure "uniform ergodicity/properties" by avoiding "forbidden" values of $\theta$?
				- I think so...


## AM - Haario 
empirical (online) estimate of the $\pi$ covariance is used and scaled by 2.38^2 (which has some theoretical justification when using Gaussian proposals in Metropolis updates)

## MwG - Roberts (probably)

## (SR)RWM - Andrieu (and everyone else, really)


we tend to centre our mean around the prev sample $X_i$ (obvs) but don't make the (co-)variances depend on the current sample, BUT we can!
	- (and keep track of/tune this dependency based on the acceptance rate)

## Langevin - Atchade
according to Roberts' slides: use proposal $\mathcal{N}(X + \vargamma \grad \log \pi(X)/2, \vargamma)$.
	- (Not entirely sure what that ^ means...)




## Low-rank updates
just update one or a few of the dimensions at a time with metropolis (or e.g. MwG) low-dim update:
	- could do each dim independently (equivalent to full-dim RWM w/ diagonal covariance matrix)
	- could do only a few/a single dims at each iteration:
		- dims chosen at random? Uniformly or according to some idea of importance
		- could choose this based on (online!) PCA

but then we've also got to consider how best to update each of the dims:
	- use the empirical variance approach?
	- try to obtain a per-component acceptance rate of 0.44 (for AMwG)
	- something else?

In general, Roberts seems to favour MwG over everything else

## Langevin 
more efficient (generally speaking) tha RWM and MwG btu less robust to difficult problens (e.g. light tails, discontinuous targets ($\pi$?))

## Wang and Landau algorithm - adapted (har har) by Atachde and Liu
seems complicated

but the paper https://www3.stat.sinica.edu.tw/sstest/oldpdf/A20n16.pdf#cite.andrieuetrobert02 has a nice introduction section that lists some of the most important work in adaptive mcmc


## Massively Parallel MCMC
**WE** have an important restriction on our proposal in MP-land: 
	"the prob density should be the same for any choice of $k_i$ and $k_i' \neq k_i$",
 i.e. the $K$ samples of each latent should receive the same proposal, but perhaps centered around themselves (i think??)

AND ALSO: 
	"we consider a proposal which is independent of other latent variables"

Qs:
	1. is that "i think??" above ^ true?
	2. Can the proposal depend on OLD samples of other latent variables?
	2a. Can the proposal actually depend on other latents w/out problem? is the reason we're avoiding this b/c stuff just gets complicated?
