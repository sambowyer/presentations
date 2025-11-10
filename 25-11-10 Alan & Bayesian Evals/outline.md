# Cohere Interview Presentation - Alan & Bayesian Evals
(40 mins + 5 mins Q&A)

## Overview
- One slide about me
- 15 mins on alan
- 25 mins on bayesian evals
- 5 mins Q&A

## Me
- Fourth (final) year PhD student at Bristol's Compass CDT
- Working on discrete diffusion models
- Two projects I'll talk about today: Alan & Bayesian Evals

## Alan
- Overview:
    - Work with Laurence and Thomas
    - A probabilistic programming language in pytorch that makes use of GPU acceleration for inference over general, user-specified models
    - Based on 'massively parallel' Bayesian inference
- Massively parallel Bayesian inference:
    - Regular Bayesian inference -- importance weighting
    - Global IS scales poorly
    - Massive parallelisation -- reason about all $K^N$ possible joint samples at once
    - Use figure from poster to explain
- GPU speedup
    - Put graphical model horizontal and show factorisation over variables
    - Reduces to tensor multiplication (good for GPUs!)
- VI:
    - Compare global VI objective and MP VI objective
    - Show ELBO go up on some datasets
    - Show graphical model for movielens?
    - Cite TMC
- Autodiff for posterior sampling:
    - Get posterior sample estimates using moment-generating function
    - Cite MOM paper
- Use these in an adadptive IS algorithm (QEM)
    - Show some plots of QEM vs global VI
    - Cite QEM paper
- Pytorch interface



## Bayesian Evals
- Overview:
    - Work with Laurence and Desi
    - Wanted to do Bayesian evaluation of LLMs
    - Two directions:
        - Interpretability of evals with SAE-like things
        - Improved UQ for evals
- Position: Don't Use the CLT...
    - Current practice: Use the CLT... bad bc x,y,z &c.
    - Mention ICML spotlight 
- Other IID methods:
    - Beta-Bernoulli
        - NOTE: credible interval NOT confidence interval
    - Wilson
    - CLopper-Pearson
    - Bootstrap
- Simulation setup
- Results
    - Explain metrics
- Looking past IID...
- Subtasks/clustering
    - E.g. anthropic clustered std err
    - Bayesian modelling
    - Results
- Comparing two models:
    - Independent
    - Paired
    - ... use this as an argument for bayes over freq
- Also:
    - Other metrics (F1)
    - Prior mismatch
- Simple package for practitioners to use `bayes_evals`
- Interpretability of evals:
    - Do modelling of evals more generally
    - Show SAE plots
    - "We didn't end up taking this forward as we weren't getting a rich enough signal"
    - "... however, I think looking at it as a predictive model for pretraining is a really interesting direction"
        - (This is something I talked briefly to Kris about at ICML)
    - "(which we didn't have the resources to do)"
    