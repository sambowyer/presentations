# Presentations

A collection of presentations on various ML subjects/papers.
Mostly for internal lab meetings.

## 2022
- [HMMs (Bootcamp)](<22-09-07 HMMs (Bootcamp)>)
    - An introductory presentation on Hidden Markov Models and the forward-backward and Baum-Welch algorithms.
    - _Presentation for [Compass CDT](https://www.bristol.ac.uk/cdt/compass/) first-year Bootcamp module_

## 2023
- [Self-Tuning Spectral Clustering](<23-02-24 Self-Tuning Spectral Clustering>)
    - Discusses the paper [Self-Tuning Spectral Clustering](https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf) which introduces a novel spectral clustering algorithm that automatically tunes the number of clusters and kernel bandwidth.
    - _Presentation for [Compass CDT](https://www.bristol.ac.uk/cdt/compass/) first-year Statistical Methods 2 module_
- [Audio Diffusion](<23-05-02 Audio Diffusion>)
    - Diffusion-based neural vocoders for high-quality audio generation, with a focus on [WaveGrad](https://arxiv.org/abs/2009.00713) and [DiffWave](https://arxiv.org/abs/2009.09761).
    - _Lab meeting presentation_
- [Adaptive MCMC (mini)](<23-06-15 Adaptive MCMC (mini)>)
    - Introduction to Adaptive Markov Chain Monte Carlo methods for efficient sampling.
    - _Internal presentation_
- [AMP-IS Exploration](<23-08-02 AMP-IS Exploration>)
    - Discusses the AMP-IS algorithm (later renamed QEM) for efficient massively-parallel Bayesian inference with [alan](https://github.com/alan-ppl/alan).
    - _Internal presentation_
- [VOD (Variational Open-Domain Q&A)](<23-08-08 VOD (Variational Open-Domain Q&A)>)
    - Presents the paper [Variational Open-Domain Question Answering](https://arxiv.org/pdf/2210.06345) which uses Rényi divergence variational inference for RAG in open-domain question answering.
    - _Lab meeting presentation_
- [ML vs RWS](<23-08-09 ML vs RWS>)
    - Compares QEM (AMP-IS) and [(massively parallel) Reweighted Wake-Sleep (RWS)](https://arxiv.org/pdf/2305.11022) for Bayesian inference.
    - _Internal presentation_
- [SBI](<23-11-07 SBI>)
    - An overview of neural simulation-based inference methods. Largely based on the paper [The frontier of simulation-based inference](https://arxiv.org/pdf/1911.01429).
    - _Lab meeting presentation_

## 2024
- [Sequential Bayes for Continual Learning](<24-02-20 Sequential Bayes for Continual Learning>)
    - Discusses the use of sequential Bayesian inference in continual learning scenarios, specifically via the [ProtoCL](https://arxiv.org/abs/2301.01828) algorithm.
    - _Lab meeting presentation_
- [KANs (Kolmogorov-Arnold Networks)](<24-05-28 KANs (Kolmogorov-Arnold Networks)>)
    - Presents [Kolmogorov-Arnold Networks (KANs)](https://arxiv.org/abs/2404.19756), a novel neural network architecture to rival MLPs.
    - _Lab meeting presentation_
- [MP-VI](<24-07-05 MP-VI>)
    - An introduction to massively parallel inference (see: [alan](https://github.com/alan-ppl/alan) and [Tensor Monte Carlo](https://arxiv.org/abs/1806.08593)) and variational inference more generally.
    - _Compass Seminar presentation_
- [Transformer Neural Processes](<24-07-30 Transformer Neural Processes>)
    - Covers the paper [Transformer Neural Processes: Uncertainty-Aware Meta Learning Via Sequence Modeling](https://arxiv.org/abs/2207.04179).
    - _Lab meeting presentation_
- [Why Linearised Laplace is Better](<24-10-08 Why Linearised Laplace is Better>)
    - Covers the paper [Reparameterization invariance in approximate Bayesian inference](https://arxiv.org/abs/2406.03334) which explains why the massive number of reparameterisations available in Baysian Neural Networks leads to the Linearised Laplace approximation underfitting less than the Laplace approximation. The paper also introduces 'Laplace Diffusion' as a reparameterisation-invariant approximation for non-linearised BNNs.
    - _Lab meeting presentation_
- [Finetuning LLMs](<24-10-22 Finetuning LLMs>)
    - A short talk entitled _"How YOU -- yes, YOU! -- can train an LLM"_ which covers the basics of finetuning large language models, specifically using LoRA, and why a Bayesian approach might be useful.
    - _Lightning talk at the 2024 [Compass Conference](https://www.bristol.ac.uk/cdt/compass/compass-annual-conference/)_
