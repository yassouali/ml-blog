---
title: "Deep Semi-Supervised Learning"
excerpt: "Given the large amounts of training data required to train deep nets,
          but collecting big detasets is not cost nor time effective. As a result 
          there is a growing need to develop data efficient methods.
          Semi-supservised learning (SSL) is possible solutions to such hurdles. In this blog post
          we present some of the new advance in SSL in the age of Deep Learning."
date: 2020-01-17 18:00:00
published: false
tags: 
  - long-read
  - deep-learning
  - semi-supervised
---

Deep neural networks demonstrated their ability to provide 
remarkable perfomances on certain supervised learning tasks (e.g. image classification, language modeling) when trained on
large collections of labeled data (e.g. ImageNet, Wikitext-103). However,
creating such large collections of data requires a considerable amount of ressources, time and effort. Such resources may not be available in a lot of practical cases,
which limits the adoption and application of many deep learning (DL) methods.

In search for more data efficient DL methods to overcome the need for large
annotated datasets, we are seeing a lot
of research interest in recent years with regards to the application of semi-supervised learning (SSL) to deep neural nets as a possible alternative,
by developping novel methods and adopting existing one for a
deep learning setting
This post discusses SSL in a deep learning setting, and goes through some of SSL main methods.

# Semi-supervised Learning

## What is Semi-supervised Learning?

> Semi-supervised learning (SSL) is halfway between supervised and unsupervised learning.
In addition to unlabeled data, the algorithm is provided with some supervision
information – but not necessarily for all examples. Often, this information will
be the targets associated with some of the examples. In this case, the data set \\( X=\left(x_{i}\right); i \in [n]\\)
can be divided into two parts: the points \\( X_{l}:=\left(x_{1}, \dots, x_{l}\right) \\), for which labels
\\( Y_{l}:=\left(y_{1}, \dots, y_{l}\right) \\) are provided, and the points
\\( X_{u}:=\left(x_{l+1}, \ldots, x_{l+u}\right) \\), the labels of which are
not known.
> <footer><strong>Chapelle et al.</strong> &mdash;
> <a href="http://www.acad.bg/ebook/ml/MITPress- SemiSupervised Learning.pdf">SSL book</a> 
> </footer>

As stated in the definition above, in SSL, we are provided with a dataset containing both
labeled and unlabeled examples. The portion of labeld examples is usually quite small
compared to the unlabeld example (e.g. 1 to 10% of the total number of examples). So with a
dataset \\(\mathcal{D}\\) containing a labeld subset \\(\mathcal{D_l}\\) and an unlabeled subset
\\(\mathcal{D_{ul}}\\). The objective, or rather hope, is to laverage the unlabeled
examples to train a model and obtain better performance than what can be obtained using only the
labeled portion. And hopefully, get closer to the desired optimal perfomance, in which
all of the dataset \\(\mathcal{D}\\) is labeled.

More formally, the goal of SSL is to
leverage the unlabeld data \\(\mathcal{D_{ul}}\\) to produce a prediction function
\\(f_{\theta}(x) \\) with trainable parameters \\(\theta\\), that is more accurate than what would have been obtained by only using the labeled data \\(\mathcal{D_l}\\).
For instance, \\(\mathcal{D_{ul}}\\) might provide us with additionnal infomation about the structure of the data distribution \\(p(x)\\), to better estimate the decision boudary
between the different classes. As shown in Fig. 1 bellow, where the data points with distinct labels are seperated with low density regions, laveraging unlabled data with SSL approach can provid us with additionnal information about the shape of the decision boundary between two classes and reduce the ambuiguity present in the supervised case.

<figure style="width: 75%" class="align-center">
  <img src="{{ 'images/SSL/cluster_ssl.png' | absolute_url }}" alt="">
  <figcaption>Fig. 1. The decision boundaries obtained on two moons dataset, with a supervised and diffrent SSL approaches, using 6 labeled examples, 3 for each class and the rest of the points as unlabeld data. (Image source: <a href="https://arxiv.org/abs/1804.09170">Oliver et al</a>)
  </figcaption>
</figure>

Semi-supervised learning first appeared in the form of self-training, where a model is  first trained on labeled data and then, iteratively, at each training iteration, a portion of the unlabeled data is annotated using the trained model and added to the training set for the next iteration. SSL really took of in 1970s after its sucess
with iterative algorithm such as the [expectation-maximization](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm) algorithm using labeled and unlabeled
data to maximize the likelihood of the model. In this post, we are only interested in SSL in a deep learning, for a detailed review of the field, [Semi-Supervised Learning Book](http://www.acad.bg/ebook/ml/MITPress- SemiSupervised Learning.pdf) is good resource.

## Semi-supervised learning methods

There have been many SSL methods and approches that have introduced over the years, SSL algorithms can be broadly divided into the following categories:

- **Consistency Regularization (Consistency Training).** Based on the assumption that if a realistic perturbation was applied to the unbeled data points, the prediction should not change significantly. We can then train the model to have a consistent prediction on a given unbaled example and its perturbed version.
- **Generative models.** Similar to the supervised setting, where the learned features on one task can be transferred to other down stream tasks. Generative models that are able to generate images from the data distribution \\(p(x)\\) must learn transferable features to a supervised task on \\(p(x \| y)\\) for a given task with targets \\(y\\).
- **Graph Based Algorithms.** A labeld and unbaled data points constitute the nodes of the graph, and the objective is to propagate the labels from the labeled nodes to the unlabled ones. The similarity of two nodes \\(n_i\\) and \\(n_j\\) is reflected by how strong is the edge \\(e_{ij}\\) between them.
- **Bootstraping.** A trained model on the labeled set can be used to produce additionnal training examples extracted from the unlabled set, the extracted examples can be based on some heuristic. Some examples of Bootstraping based SSL are *Self-training*, *Co-training* and *Multi-View Learning*.

In addition to these main categories, their is also some SSL work on **entropy minimization**, where we force the model to make confident preditions by minimizing the entropy of the predicitons. And **pseudo-labels** where a trained model on the labeled data is utilized to label the unbaled portion and the new targets are used in a standard supervised setting.

In this post, we will focus more on consistency regularization based approches, given that they are the most commonly used methods in deep learning, and we will present a brief introduction to the other methods, 

## Assumptions:

But when can we apply SSL algorithms? SSL algorithms only work under some conditions, and follow some assumptions about the structure of the data that need to hold. Without such assumptions, it would not be possible to generalize from a finite training set to a set of possibly infinitely many unseen test cases.

The main assumptions in SSL are:
* **The Smoothness Assumption**: *« If two points \\(x_1\\), \\(x_2\\) in a high-density regionare close, then so should be the corresponding outputs \\(y_1\\), \\(y_2\\) »*. Meaning that if two inputs are of the same class and belong to the same cluster, which is a high desity region of the input space, the their correponding outputs need to be close. And the inverse hold true (if the two points are separated by a low-density region, the outputs must distant). This assumption can be quite helpful in a classification task, but not so much for regression.
* **The Cluster Assumption**: *« If points are in the same cluster, they are likely to be of the same class. »* In this case, we suppose that input data points form clusters, and each cluster correponds to one of the output classes. And  decision boudary must lie in low density region for get the correct classification. This assumption is a special case of the smoothness assumption. With this assumption, we can restrict our model to have consistent prediction on the unblaled data over some small perturbations.
* **The Manifold Assumption**: *« The (high-dimensional) data lie (roughly) on a low-dimensional manifold. »* With high dimentionnal space, where the volume grow exponenetially with the number of dimsions, it can be quite hard to estimate the true data distribution for generative tasks. For discriminative tasks, the distances are similar regardless of the class type, make classification quite chalenging. However, if our input data lies on some lower dimensionnal manifold, we can try to find low dimensionnal representation using the unlabled data, and then use the labled data to solve the simplefied task.

# Consistency Regularization

A recent line of work in deep semi-supervised learning is utilizing unlabled data
to enforce the trained model to be inline with the cluster assumption, i.e., the
learned decision boundary must lie in low density regions. These methods are based
on a simple concept, if a realistic perturbation was to be applied to an unlabeled
example, the predictions should not change significantly. Given that under the
cluster assumption, data points with distinct labels are sperated with low density regions,
so the likelihood of one example to switch classes after a perturbation is small (see Figure 1).

More formally, with consistency regularization, we are favoring the functions \\(f_\theta\\) that give consistent
prediction for similar data points. So rather that minimizing the classification cost at the zero-dimensional data points
of the inputs space. The regularized model minimizes the cost on a manifold around each data point, pushing the
decision boundaries away from the labeled data points and smoothing the manifold on which the data
resides ([Zhu, 2005](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf)).
Given an unlaled data point \\(x_u \in \mathcal{D_u}\\) and its perturbed version \\(\hat{x}_u\\),
the objective is to minimize the distance between the two outputs
$$d(f_{\theta}(x_u), f_{\theta}(\hat{x}_u))$$. The popular distance measures $$d$$ are
mean squared error (MSE), Kullback-Leiber divergence (KL)
and Jensen-Shannon divergence (JS). For two
outputs $$y_u = f_{\theta}(x_u)$$ and $$\hat{y}_u = f_{\theta}(\hat{x}_u)$$,
and $$m=\frac{1}{2}(y_u+\hat{y}_u)$$, we can compute these measures as follows:

$$\small d_{\mathrm{MSE}}(y_u, \hat{y}_u)=\frac{1}{N} \sum_{i}^{N}(y_u(i)-\hat{y}_u(i))^{2}$$

$$\small d_{\mathrm{KL}}(y_u, \hat{y}_u)=\frac{1}{N} \sum_{i}^{N} y_u(i) \log \frac{y_u(i)}{\hat{y}_u(i)}$$

$$\small d_{\mathrm{JS}}(y_u, \hat{y}_u)=\frac{1}{2}
\mathbf{d}_{\mathrm{KL}}(y_u, m)+\frac{1}{2} \mathrm{d}_{\mathrm{KL}}(\hat{y}_u, m)$$

Note that we can also enforce a consistency over two perturbed versions of $$x_u$$,
$$\hat{x}_{u_1}$$ and $$\hat{x}_{u_2}$$. Now let's go through the popular consistency regularization methods
in deep learning.

$$

## Ladder Networks
With the objectif to take any well performing feed-forward network on supervised data and augment it with
addtionnal branches to be able to utilize additionnal unlabled data.
[Rasmus et al](https://arxiv.org/abs/1507.02672) proposed to use Ladder
Networks ([Harri Valpola](https://arxiv.org/abs/1411.7783)) consisting of an additionnal
encoder and decoder for SSL.
As illustrated in Figure 2, the network consists of two encoders, a corruped and clean one, and a decoder.
At each training iteration, and input $$x$$ is passed throught both encoder. In the corruped encoder, 
a gaussian noise is injected at each layers after batch normalization. Producing two outputs, a clean prediction
$$y$$ and a prediction based on corruped activations $$\tilde{y}$$. The output $$\tilde{y}$$ is then fed into
the decoder to reconstruct the uncorrupted input hidden activations.
The unsupervised training loss $$\mathcal{L}_u$$
is then computed as the MSE between the activations of the clean encoder $$\mathbf{z}$$
and the reconstructed activations $$\hat{\mathbf{z}}$$ (ie., after batch normalization)
in the decoder using the corrupted output $$\tilde{y}$$, this is computed over all layers,
from the input to the last layer $$L$$, with a weighting $$\lambda_{l}$$ for each layer's contribution loss:

$$\mathcal{L}_u =\sum_{l=0}^{L} \lambda_{l}\|\mathbf{z}^{(l)}-\hat{\mathbf{z}}^{(l)}\|^{2}$$

If the input $$x$$ in a labeled data point $$x \in \mathcal{D_l}$$. Then we can add a supervised loss term
to $$\mathcal{L}_u$$ to obtain the final loss. Note the supervised cross-entropy
loss is computed between the corrputed output $$\tilde{y}$$ and the targets $$t$$:

$$\mathcal{L} = \mathcal{L}_u  + \mathcal{L}_s = \mathcal{L}_u + \log P(\tilde{\mathbf{y}}|t)$$

<figure style="width: 75%" class="align-center">
  <img src="{{ 'images/SSL/ladder_network.png' | absolute_url }}" alt="">
  <figcaption>Fig. 2. An illustration of one forward pass of Ladder Networks, C refers
  to the MSE loss between the activations at various layers.
  (Image source: <a href="https://arxiv.org/abs/1507.02672">Rasmus et al</a>)
  </figcaption>
</figure>

The method can be easily addaped to convolutional neural networks (CNNs)
by replacing the fully-connected layers with
convolutionnal and deconvolutionnal layers for semi-supervised computer vision tasks.
However, the ladder network are quite heavy computationnaly, approximately tripling
the computation needed for one training iteration. To mitigate this,
the authors propose a variant of ladder networks called **Γ-Model** where
$$\lambda_{l}=0$$ when $$l<L$$. In this case the decoder is ommitted and the unsupervised loss
is computed as the MSE between the two outputs $$y$$ and $$\tilde{y}$$.

$$

## Π-model 
The **Π-model** ([Laine et al](https://arxiv.org/abs/1610.02242)) is a simplification of the **Γ-Model** of Ladded Networks,
where the denoising encoder is removed and the same network is used to get the prediction for both corrupted and uncorrupted inputs.
Specifically, **Π-model** takes advantage of the stochastic nature of the prediction function $$f_ \theta$$ in neural network due to common regularization techniques, such as data augmentation and dropout that typically don't alter the model's predictions.
For any given input $$x$$, the objective is to reduce the distances between two predictions of $$f_ \theta$$ with $$x$$ as input in both forward passes.
Concretly, as illustrated in Figure 3, we would like to minimize $$d(\hat{z}_1, \hat{z}_2)$$. Given the stochasitc nature of the predictions function (ie., using dropout as noise source), the two outputs $$\hat{z}_{1} = f_\theta(x)$$ and $$\hat{z}_2 = f_\theta(x)$$ will be distinct. And the objective is to obtain consistent predictions for both of them. In case the input $$x$$ is a labled data point, we also compute the cross-entorpy supervised loss using the provided labels $$y$$ and the total loss will be:

$$\mathcal{L} = w(t)\ d_{\mathrm{MSE}}(\hat{z}_1, \hat{z}_2) + y\log(z)$$

With $$w(t)$$ as a weighting function, starting from 0 up to a fixed weight $$\lambda$$ (eg., 30) after a given number of epochs (eg., 20% of training time). This way, we avoid using the untrained and random prediction function providing us with unstable predictions at the start of training to extract the training signal from
the unlabeled examples.

<figure style="width: 100%" class="align-center">
  <img src="{{ 'images/SSL/pi_model.png' | absolute_url }}" alt="">
  <figcaption>Fig. 3. Loss computation for <b>Π-model</b>, we compute the MSE between the two outputs, and if the inputs
  is a labeled we add the supervised loss to the weighted unsupervised loss.
  (Image source: <a href="https://arxiv.org/abs/1610.02242">Laine et al</a>)
  </figcaption>
</figure>




### References

[1] Chapelle et al. [Semi-supervised learning book](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf). IEEE Transactions on Neural Networks, 2009.

[2] Xiaojin Jerry Zhu. [Semi-supervised learning literature survey](http://www.acad.bg/ebook/ml/MITPress- SemiSupervised Learning.pdf). Technical report, University of Wisconsin-Madison Department of Computer Sciences, 2005.

[3] Rasmus et al. [Semi-supervised learning with ladder networks](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf). Advances in neural information processing systems, 2015.

[4] Samuli Laine, Timo Aila. [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242). ICLR 2017.

[5] Harri Valpola [From neural PCA to deep unsupervised learning](https://arxiv.org/abs/1411.7783). Advances in Independent Component Analysis and Learning Machines 2015.