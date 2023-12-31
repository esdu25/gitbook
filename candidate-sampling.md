# Candidate Sampling

The Candidate Sampling doc from [https://www.tensorflow.org/extras/candidate\_sampling.pdf](https://www.tensorflow.org/extras/candidate\_sampling.pdf) covers many important concepts. Here are my reading notes.

## Sampled Softmax

### Compatibility Function F(x,y) is the Logit

"Compatibility Function" $$F(x,y)$$ is defined as:

$$
F(x,y) \leftarrow \log P(y|x) + K(x)
$$

We can derive this from Softmax:

$$
\begin{align*}
P(y|x) &= \frac{e^z}{\sum_j e^{z_j}} \\
\log P(y|x) &= \log \frac{e^z}{\sum_j e^{z_j}} \\
\log P(y|x) &= \log e^z - \log \sum_j e^{z_j} \\
\log P(y|x) &= z - \log \sum_j e^{z_j} \\
z &= \log P(y|x) + \log \sum_j e^{z_j} \\
z &= \log P(y|x) + K(x) \\
\end{align*}
$$

Hence, $$F(x,y)$$ is, as the doc says, the "softmax logits", the unbounded real-number output from a neural network.

Interestingly, the logit can be viewed as a "log-prob + a constant wrt. y".

To double check, we plug it back into the softmax:

$$
\begin{align*}
\frac{e^z}{\sum_j e^{z_j}} 
&= \frac{e^{\log P(y|x) + K(x)}}{\sum_j e^{\log P(y_j|x) + K(x)}}  \\
&= \frac{P(y|x) e^{K(x)}}{\sum_j P(y_j|x) e^{K(x)}}  \\
&= \frac{P(y|x) }{\sum_j P(y_j|x) }  \\
&= \frac{P(y|x) }{1}  \\
&= P(y|x) 
\end{align*}
$$

### What changed from Full Softmax to Sampled Softmax

Due to the inability to compute the denominator exactly in Full Softmax, we opt to estimate the Full Softmax (the probability) via a sample instead, i.e. a Monte Carlo estimate.

The Full Softmax's distribution of data is _uniform,_ but we assume the sampling distribution is _any distribution_; it could be non-uniform.

Specifically, we assume each item is sampled according to a Bernoulli distribution with PMF $$Q$$. For the i'th training example, we take a sample $$S_i$$, and the probability we can pick this particular sample is $$P(S_i = S | x_i) = \prod_{y\in S}Q(y|x_i) \prod_{y\in(L-S)}(1-Q(y|x_i))$$

We combine the sample and the original training label to form the Candidates Set $$C_i = S_i \cup \{t_i\}$$

Note the change:

* Full Softmax := $$P(y|x_i)$$, i.e. pick the right item out of the _universe of items._
* Sampled Softmax := $$P(y|x_i,C_i)$$, i.e. pick the right item out of the _candidate items._

Our goal is to compute $$P(y|x_i,C_i)$$ using $$P(y|x_i)$$.

> Note: This is only useful during training time. At inference time we can simply use $$P(y|x_i)$$

What changed is the addition of $$C_i$$.&#x20;

$$
\begin{align*}
P(t_i = y | C_i, x_i) &= \frac{P(t_i = y, C_i | x_i)}{P(C_i | x_i)} \\
&= \frac{P(t_i = y | x_i) P(C_i | t_i=y, x_i)}{P(C_i | x_i)} \\
&= \frac{P(y | x_i) P(C_i | t_i=y, x_i)}{P(C_i | x_i)}
\end{align*}
$$

We use Bayes rule; $$P(y|x_i)$$ is the prior, $$P(y|C_i,x_i)$$ is the posterior.

Note the importance of $$t_i = y$$ , instead of just writing $$y$$, is to highlight that $$y$$ is a part of $$C_i$$, therefore when we compute $$P(C_i|t_i=y,x_i)$$, we can leave out the $$y$$ element.

$$
\begin{align*}
P(t_i = y | C_i, x_i) &= \frac{P(y | x_i) P(C_i | t_i=y, x_i)}{P(C_i | x_i)} \\
&= \frac{P(y | x_i) \prod_{y\in C_i - \{y\}} Q(y|x_i) \prod_{y\in (L-C_i)} (1-Q(y|x_i)) }{P(C_i | x_i)} \\
&= \frac{P(y | x_i) \frac{Q(y|x_i)}{Q(y|x_i)} \prod_{y\in C_i - \{y\}} Q(y|x_i) \prod_{y\in (L-C_i)} (1-Q(y|x_i)) }{P(C_i | x_i)} \\
&= \frac{\frac{P(y | x_i)}{Q(y|x_i)} \prod_{y\in C_i} Q(y|x_i) \prod_{y\in (L-C_i)} (1-Q(y|x_i)) }{P(C_i | x_i)} \\
&= \frac{P(y | x_i)}{Q(y|x_i)} K(x_i,C_i) \\
\log P(t_i = y | C_i, x_i) &= \log P(y | x_i) - \log Q(y|x_i) + \log K(x_i,C_i)
\end{align*}
$$

<figure><img src=".gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

### LogQ Correction

During training, since we are using a sampled denominator, we use cross-entropy loss on the corrected probability, that is: $$H(y,\hat{y}) = -\sum P(y_i) \log(P(\hat{y_i})) = -\sum P(y_i) \log(P(t_i = y_i | x_i, C_i))$$.

Recall the logit $$F(x,y) =\log P(y|x) + K(x)$$

Therefore the new logit is:

$$
\begin{align*}
\log P(t_i = y | C_i, x_i) &= \log P(y | x_i) - \log Q(y|x_i) + \log K(x_i,C_i) \\
 &= F(x,y) - \log Q(y|x_i) + (\log K(x_i,C_i) - \log K'(x_i,C_i)) \\
 &= F(x,y) - \log Q(y|x_i)
\end{align*}
$$

The last line is true because $$K$$is a constant w.r.t. $$x_i$$, hence if we took the softmax, it gets cancelled out

$$
\frac{e^{z + K}}{\sum e^{z_i + K}} = \frac{e^K e^{z}}{e^K\sum e^{z_i}} = \frac{e^{z}}{\sum e^{z_i}}
$$

Note that if Q is uniform (i.e. it matches the Full Softmax distribution), then it gets merged into $$K$$ and therefore has no effect.

## Related Material

* [https://github.com/tensorflow/recommenders/issues/257](https://github.com/tensorflow/recommenders/issues/257)
  * Very well explained
* "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" [https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/6417b9a68bd77033d65e431bdba855563066dc8c.pdf](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/6417b9a68bd77033d65e431bdba855563066dc8c.pdf)&#x20;
  * Uses Sampled Softmax + LogQ correction in the YouTube recs retrieval. Training data comes from real user activities, so Q is not static; new videos get uploaded, and popularity of videos change. **The authors propose an online algorithm to estimate Q(y) with streaming data**.
* "Adaptive Importance Sampling to Accelerate Training of a Neural Probabilistic Language Model", 2007 [https://www.iro.umontreal.ca/\~lisa/pointeurs/importance\_samplingIEEEtnn.pdf](https://www.iro.umontreal.ca/\~lisa/pointeurs/importance\_samplingIEEEtnn.pdf)
  * Bengio's work on estimating the partition function with a sampling distribution. However, he says the estimator is biased... didn't get a chance to look in depth.
* Not read
  * "Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations" [https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/b9f4e78a8830fe5afcf2f0452862fb3c0d6584ea.pdf](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/b9f4e78a8830fe5afcf2f0452862fb3c0d6584ea.pdf)
  * "On Using Very Large Target Vocabulary for Neural Machine Translation", 2015 [https://arxiv.org/pdf/1412.2007.pdf](https://arxiv.org/pdf/1412.2007.pdf)&#x20;
    * Another commonly cited paper



