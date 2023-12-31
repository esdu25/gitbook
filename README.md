# LogQ Correction

In RecSys, it's common to only have positive labels as training data, e.g. a stream of "click"s as (context, item) pairs.

(This is actually an old problem in NLP, where a word and its context is a positive pair)

We need negatives labels, e.g. (context, item\_neg), to train the model. Ideally, the negatives should be sampled from the universe of items **uniformly**.

There are two popular ways to get negatives:

1. Sample K items uniformly from the universe of items and pretend they are negatives.
2. Use the other K items in the batch as negatives.

### Method 1

This method is used to train word2vec, 2013.

For each positive label in the batch (context, item\_pos), we uniformly sample K negative labels (context, item\_neg\_i) ... (context, item\_neg\_k).

We view this as a **set of binary classification problems**; use a logistic regression loss (BCE) for each label, and sum all the losses together in a batch.

This is effectively a Monte Carlo estimate of $$BCE(x,y^{pos}) - E_{unif}[BCE(x,y_{i}^{neg})]$$

(The $$x$$ is sigmoid(nn(context, item)),  and $$y$$is either 0 or 1)

Note that <mark style="background-color:red;">this doesn't address the problem that popular items appear more often as positive labels.</mark>

### Method 2: Sampled Softmax with LogQ Correction

This method uses all other elements in the batch as negatives. This can be easier to implement since it avoids explicitly sampling negatives in the data pipeline.

First, let's look at if we could use the entire dataset as the batch on each update.

We formulate it as "pulling up" the probability of the positive label, while implicitly "pushing down" the probability of negative labels.

For item $$y$$ , context $$x$$, model $$s$$ with parameters $$\theta$$ that outputs a logit, and $$N$$ items in the universe. We can define the probability of an item getting recommended, given a context,&#x20;

$$
P(y | x ; \theta) = \frac{e^{s_\theta(x, y)}}{\sum_{j=1}^N e^{s_\theta(x,y_j)}}
$$

We thus have the likelihood function, for all training pairs in training set $$T$$ :

$$
L(\theta) = \prod_{x,y}^T P(y|x ; \theta)
$$



Maxinimizing the likelihood = Minimizing the negative log-likelihood:

$$
\begin{align*} 
-\ln(L(\theta)) &= -\sum_{x,y}^T \ln P(y|x; \theta) \\
&= -\sum_i^T [\ln \frac{e^{s_\theta(x, y)}}{\sum_{j=1}^N e^{s_\theta(x,y_j)}}]
\\
&= -\sum_i^T [\ln (e^{s_\theta(x, y)}) - \ln (\sum_{j=1}^N e^{s_\theta(x,y_j)})]
\\
&= -\sum_i^T [s_\theta(x, y) - \ln (\sum_{j=1}^N e^{s_\theta(x,y_j)})]
\\
&= \sum_i^T [\ln (\sum_{j=1}^N e^{s_\theta(x,y_j)}) - s_\theta(x, y)]
\end{align*}
$$

The log-sum-exp is too expensive to calculate on every update, since the number of items $$N$$ can be in the millions or billions.

We could, instead, take a uniform Monte Carlo estimate of this sum, which results in summing over a batch of items $$B$$:&#x20;

$$
-\ln(L(\theta)) \approx \sum_i^T [\ln (\sum_{j=1}^B e^{s_\theta(x,y_j)}) - s_\theta(x, y)]
$$

However, in practice, the batches we sample do not come from a uniform sample of items. They come from a "sampling distribution" $$Q$$ of real user interactions. Under $$Q$$ , more popular items are more likely to appear as negatives (and positives.)

That is, in practice the batch sum is a Monte Carlo estimate of:

$$
\sum_{j=1}^B e^{z_j} \approx B \cdot E_Q[e^z] = B \cdot  \sum_{j=1}^N Q(z_j)e^{z_j}
$$

We can correct this back to a uniform sample with Importance Sampling:

$$
B \cdot \sum_{j=1}^N P(z_j) \frac{Q(z_j)}{P(z_j)} e^{z_j} = B \cdot E_P [\frac{Q(z)}{P(z)}e^z]
$$









* We're not correcting for the positive also appearing too many times?





$$f(x) = x * e^{2 pi i \xi x}$$





* [https://www.tensorflow.org/extras/candidate\_sampling.pdf](https://www.tensorflow.org/extras/candidate\_sampling.pdf)
*
