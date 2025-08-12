

"we hypothesize two reasons for why fine-tuning foundation models using offline RL is unsuitable in practice:
1. First, typical offline RL methods require regressing value functions that estimate how appropriate actions, such as an utterance in dialogue" (Q-learning) [...] These scale badly (instability in the value-learning objective?)
2. Q-learning aims to predict action-values which ignores the learned likelihood information from (supervised) pretraining 

"Our key insight is simple: by adding weights to the traditional supervised fine-tuning objective, we can learn probabilities that conservatively estimate the value function instead of the behavior policy"

-> construct "weighted cross-entropy loss where weights are target action values computed from the Bellman recurrence relations"

**"instead of training value functions by fitting Q-values to their Bellman backup target via a regression loss, we instead fine-tune directly on the probabilities learned from large-scale pretraining —like in SFT— via a weighted cross-entropy loss, such that the resulting probabilities also capture the desired Q-values."**

"Our goal is to provide a way to learn Q-values for multi-turn RL problems with language models such that the Q-function can be initialized from a model pretrained via supervised learning (i.e., maximum likelihood estimation), without the need to reinitialize weights or add new heads to represent the Q-values."