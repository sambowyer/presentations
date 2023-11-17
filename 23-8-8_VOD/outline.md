intro: ~2 mins
	what is ODQA
	
	authors say VOD is versatile beyond the Q&A analysis/implementation here

basic reader/receiver outline: ~5 mins
	Eq 1

	can optimise with VI (ELBO) by introducing an approx posterior
		but authors suggest they know better...

Renyi Divergence VI: ~5-10 mins
	RVB (Eq 2)

	IW RVB (Eq 3)

		+ all the nice properties therein

	Figure 2

	Gradients:
		can compute as [...probably not worth actually showing the equation, just say "easily"/"tractable"]
	
		then the observations from "Stabilizing training using the RVB"
			incl. the (dis-?)joint optimisation of reader & retriever

	look at the behaviour w/ alpha = 1 (knowledge distillation)
		vs alpha = 0 (tighter bound on true marge ll)
			...so can we use that?
				[maybe show this later? w/ Fig 3?]
				[YES, show later in VOD implementation section]

VOD objective: ~5-10 mins
	Problem: IW-RVB intractable w/ large corpora (if N large then IW needs to sum over N)
	Solution: *priority sampling*

		Priority sampling:
			basically IW (w/ a uniform proposal) and only keep the top K weights, set the rest equal to the (K-1)th weight

			and (somewhat surprisingly) we get a consistent estimator of functions out the other end! nice.

	So the VOD objective is Eq 6
		(probably not worth explaining defn of v (Eq 7)
			just say it's a priority sampling approximation of the regular importance weight)

	NOTE: O(K) rather than IW-RVB's O(N), hurrah!

	
VOD implementation: ~10 mins
	we have scoring functions f_theta and f_phi:
		f_theta is the retriever score (via BERT)
			this is only used on K documents at a time 
		f_phi is the approx posterior score (via BERT/BM25 hybrid):
			this is used on all N documents BUT r_phi only samples K from P < N of them

		-> THEN:
			p_theta is just a softmax over f_theta values
			r_phi   is just a softmax over f_phi   values

	Have your p_\theta modelled by BERT (which we're finetuning this)

	Have your r_\phi modelled by a hybrid of BERT checkpoints and BM25 ratings

	In each iteration, for a question q, with M (=4) answers a_m, we:
		1. use f_phi to score all documents, and choose the top P of them to be put in cache
		2. use r_phi to sample K out of the P documents
		3. use these P documents to compute the VOD objective (Eq 6) and its gradient wrt theta 
			N.B. This requires 
				- an f_theta forward pass for each of the K documents (O(K))
					[this also requires 1 call to encode the question (O(1))]
				- one p_theta sample to 'pick' the answer (O(1))
				[- an r_phi forward pass for each of the P documents (O(P)) but this is offline so doesn't count...]
			therefore total complexity of evaluating VOD objective is O(K) per iteration
		4. update theta

	BUT ACTUALLY, for first T iterations, we anneal \alpha from 1 to 0
		get that sweet knowledge distillation from f_phi ~= BM25 into f_theta
			because we're minimising KL(p_theta || r_phi)	
	i.e. no VOD objective stuff

	Every T iterations update your checkpoint by copying p_theta (which you've just spend T iterations updating)


	[Show Fig 3]

Experiments: ~5 mins
	Medical Q&A datasets
	
	Use the model (Eq 1) to evaluate the VOD objective on M(=4) answers and pick the most likely one (highest VOD (Eq 6) bc VOD is lower bound on true marge ll log p_theta (a|q)) 
		(actually do 10 Monte Carlo samples 
			(each of which contains MK = (4)(8) = 32 documents)
		)
	Performs well

	Better at knowledge retrieval than reasoning (see Section 4.5 "...Semantic Search")


Conclusion: ~2 mins
	seems to be good

