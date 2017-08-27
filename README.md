# CRFBiaffineParser

The gradients calculating is implemented as follow (adding a _new CRF node_ to Tensorflow to support calculating the marginal probability),

	a) Calculating the scores of the possible arcs in a sentence.
	b) Running the inside-outside algorithm to calculate the marginal probability `p(w1-->w2)` of each dependency arc.
	c) summing all `Iscorrect(w1-->w2) - p(w1-->w2)` * `Embedding node(w1-->w2)` to get the final neural node `Nf`.

This repository is built on the code of [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734). 
