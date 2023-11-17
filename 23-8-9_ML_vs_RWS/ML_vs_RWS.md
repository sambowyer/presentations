# Main points:
- Got ML2_Toy == RWS by changing toy `elbo` and `elf` computation
- BUT: no longer have ML2 = ML2_Toy or ML1 = ML1_Toy
- (...lines of code showing we're not summing over N inside anymore...)
- So if we rewrite alan model to separate out the latents and avoid this summing over $N$ ...
	- (and slightly modify the toy sampling procedure...)
- Then we get:
	- ML2 == ML2_Toy == RWS
	- ML1 == ML1_Toy
	- ML1 != ML2
- But this model specification is ugly and slow
	- ***TODO***:  actually do a quick test to compare times of different methods

__Q__: Does this indicate that we should remove 'summing over $N$' in ML2 (alan)? (`sum_non_dim` &c.)
		-> I've tried this but couldn't really get anything working
		-> Might also require us to avoid this 'summing over $N$' in the ELBO calculation (in `tensor_product()`) because the initial elbos are different between toy implementation and non-separated Alan model (but not toy vs separated-alan)
			-> also it seems unlikely to me that alan's `tensor_product()` has a bug

__Q__: OR is it just indicating that the model (and implementation of `IW()`) we're using in our toy implementation _does_ treat the latents as separated, which is something that we have to specify in alan if we want comparable/identical results?
		-> For $N=3$, the shape of the `logp` tensor corresponding to `obs` (used in `Sample.tensor_product()`) is:
			- `[K]` in non-separated alan
			-  `[K,K,K]` in separated alan
		And, whilst in non-separated alan you have one `logp` corresponding to `mu` of shape `[K]`, in non-separated alan you get $N$ `logp` tensors each of shape `[K]` and corresponding to a single latent.

My suspicion is that the latter is the case.




#### Also (really not all that important I don't think bc idk what's going on with ML1 or ML1_Toy)
- With some small $K > 1$ (in particular, $K=2$ for $N=3$ (but not $K=1$, in which case everything is the same), ML1_toy is best out of all. (This holds for all random seeds tried.)
	- And if we use separate-latent model then ML1 = ML1_toy is still best