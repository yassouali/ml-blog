---
title: "Deep Semi-Supervised Learning"
excerpt: "Given the large amounts of training data required to train deep nets,
          but collecting big datasets is not cost nor time effective. As a result 
          there is a growing need to develop data efficient methods.
          Semi-supervised learning (SSL) is possible solutions to such hurdles. In this blog post
          we present some of the new advance in SSL in the age of Deep Learning."
date: 2020-01-17 18:00:00
published: false
tags: 
  - long-read
  - deep-learning
  - semi-supervised
---

Deep neural networks demonstrated their ability to provide 
remarkable performances on certain supervised learning tasks (e.g. image classification, language modeling) when trained on
large collections of labeled data (e.g. ImageNet, Wikitext-103). However,
creating such large collections of data requires a considerable amount of resources, time and effort. Such resources may not be available in a lot of practical cases,
which limits the adoption and application of many deep learning (DL) methods.

In search for more data efficient DL methods to overcome the need for large
annotated datasets, we are seeing a lot
of research interest in recent years with regards to the application of semi-supervised learning (SSL) to deep neural nets as a possible alternative,
by developing novel methods and adopting existing one for a
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
labeled and unlabeled examples. The portion of labeled examples is usually quite small
compared to the unlabeled example (e.g. 1 to 10% of the total number of examples). So with a
dataset \\(\mathcal{D}\\) containing a labeled subset \\(\mathcal{D_l}\\) and an unlabeled subset
\\(\mathcal{D_{ul}}\\). The objective, or rather hope, is to leverage the unlabeled
examples to train a model and obtain better performance than what can be obtained using only the
labeled portion. And hopefully, get closer to the desired optimal performance, in which
all of the dataset \\(\mathcal{D}\\) is labeled.

More formally, the goal of SSL is to
leverage the unlabeled data \\(\mathcal{D_{ul}}\\) to produce a prediction function
\\(f_{\theta}(x) \\) with trainable parameters \\(\theta\\), that is more accurate than what would have been obtained by only using the labeled data \\(\mathcal{D_l}\\).
For instance, \\(\mathcal{D_{ul}}\\) might provide us with additional information about the structure of the data distribution \\(p(x)\\), to better estimate the decision boundary
between the different classes. As shown in Fig. 1 bellow, where the data points with distinct labels are separated with low density regions, leveraging unlabeled data with SSL approach can provide us with additional information about the shape of the decision boundary between two classes and reduce the ambiguity present in the supervised case.

<figure style="width: 75%" class="align-center">
  <img src="{{ 'images/SSL/cluster_ssl.png' | absolute_url }}" alt="">
  <figcaption>Fig. 1. The decision boundaries obtained on two moons dataset, with a supervised and different SSL approaches, using 6 labeled examples, 3 for each class and the rest of the points as unlabeled data. (Image source: <a href="https://arxiv.org/abs/1804.09170">Oliver et al</a>)
  </figcaption>
</figure>

Semi-supervised learning first appeared in the form of self-training, where a model is  first trained on labeled data and then, iteratively, at each training iteration, a portion of the unlabeled data is annotated using the trained model and added to the training set for the next iteration. SSL really took of in 1970s after its sucess
with iterative algorithm such as the [expectation-maximization](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm) algorithm using labeled and unlabeled
data to maximize the likelihood of the model. In this post, we are only interested in SSL in a deep learning, for a detailed review of the field, [Semi-Supervised Learning Book](http://www.acad.bg/ebook/ml/MITPress- SemiSupervised Learning.pdf) is good resource.

## Semi-supervised learning methods

There have been many SSL methods and approches that have introduced over the years, SSL algorithms can be broadly divided into the following categories:

- **Consistency Regularization (Consistency Training).** Based on the assumption that if a realistic perturbation was applied to the unlabeled data points, the prediction should not change significantly. We can then train the model to have a consistent prediction on a given unlabeled example and its perturbed version.
- **Generative models.** Similar to the supervised setting, where the learned features on one task can be transferred to other down stream tasks. Generative models that are able to generate images from the data distribution \\(p(x)\\) must learn transferable features to a supervised task on \\(p(x \| y)\\) for a given task with targets \\(y\\).
- **Graph Based Algorithms.** A labeled and unlabeled data points constitute the nodes of the graph, and the objective is to propagate the labels from the labeled nodes to the unlabeled ones. The similarity of two nodes \\(n_i\\) and \\(n_j\\) is reflected by how strong is the edge \\(e_{ij}\\) between them.
- **Bootstraping.** A trained model on the labeled set can be used to produce additional training examples extracted from the unlabeled set, the extracted examples can be based on some heuristic. Some examples of Bootstraping based SSL are *Self-training*, *Co-training* and *Multi-View Learning*.

In addition to these main categories, their is also some SSL work on **entropy minimization**, where we force the model to make confident predictions by minimizing the entropy of the predictions. And **pseudo-labels** where a trained model on the labeled data is utilized to label the unlabeled portion and the new targets are used in a standard supervised setting.

In this post, we will focus more on consistency regularization based approaches, given that they are the most commonly used methods in deep learning, and we will present a brief introduction to the other methods, 

## Assumptions:

But when can we apply SSL algorithms? SSL algorithms only work under some conditions, and follow some assumptions about the structure of the data that need to hold. Without such assumptions, it would not be possible to generalize from a finite training set to a set of possibly infinitely many unseen test cases.

The main assumptions in SSL are:
* **The Smoothness Assumption**: *« If two points \\(x_1\\), \\(x_2\\) in a high-density regions close, then so should be the corresponding outputs \\(y_1\\), \\(y_2\\) »*. Meaning that if two inputs are of the same class and belong to the same cluster, which is a high density region of the input space, then their corresponding outputs need to be close. And the inverse hold true (if the two points are separated by a low-density region, the outputs must distant). This assumption can be quite helpful in a classification task, but not so much for regression.
* **The Cluster Assumption**: *« If points are in the same cluster, they are likely to be of the same class. »* In this case, we suppose that input data points form clusters, and each cluster corresponds to one of the output classes. And  decision boundary must lie in low density region for get the correct classification. This assumption is a special case of the smoothness assumption. With this assumption, we can restrict our model to have consistent prediction on the unblaled data over some small perturbations.
* **The Manifold Assumption**: *« The (high-dimensional) data lie (roughly) on a low-dimensional manifold. »* With high dimensional space, where the volume grow exponentially with the number of dimensions, it can be quite hard to estimate the true data distribution for generative tasks. For discriminative tasks, the distances are similar regardless of the class type, make classification quite challenging. However, if our input data lies on some lower dimensional manifold, we can try to find low dimensional representation using the unlabeled data, and then use the labeled data to solve the simplified task.

# Consistency Regularization

A recent line of work in deep semi-supervised learning is utilizing unlabeled data
to enforce the trained model to be inline with the cluster assumption, i.e., the
learned decision boundary must lie in low density regions. These methods are based
on a simple concept, if a realistic perturbation was to be applied to an unlabeled
example, the predictions should not change significantly. Given that under the
cluster assumption, data points with distinct labels are separated with low density regions,
so the likelihood of one example to switch classes after a perturbation is small (see Figure 1).

More formally, with consistency regularization, we are favoring the functions \\(f_\theta\\) that give consistent
prediction for similar data points. So rather that minimizing the classification cost at the zero-dimensional data points
of the inputs space. The regularized model minimizes the cost on a manifold around each data point, pushing the
decision boundaries away from the labeled data points and smoothing the manifold on which the data
resides ([Zhu, 2005](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf)).
Given an unlabeled data point \\(x_u \in \mathcal{D_u}\\) and its perturbed version \\(\hat{x}_u\\),
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
With the objective to take any well performing feed-forward network on supervised data and augment it with
additional branches to be able to utilize additional unlabeled data.
[Rasmus et al](https://arxiv.org/abs/1507.02672) proposed to use Ladder
Networks ([Harri Valpola](https://arxiv.org/abs/1411.7783)) consisting of an additional
encoder and decoder for SSL.
As illustrated in Figure 2, the network consists of two encoders, a corrupted and clean one, and a decoder.
At each training iteration, and input $$x$$ is passed through both encoders. In the corrupted encoder, 
a gaussian noise is injected at each layers after batch normalization. Producing two outputs, a clean prediction
$$y$$ and a prediction based on corrupted activations $$\tilde{y}$$. The output $$\tilde{y}$$ is then fed into
the decoder to reconstruct the uncorrupted input hidden activations.
The unsupervised training loss $$\mathcal{L}_u$$
is then computed as the MSE between the activations of the clean encoder $$\mathbf{z}$$
and the reconstructed activations $$\hat{\mathbf{z}}$$ (ie., after batch normalization)
in the decoder using the corrupted output $$\tilde{y}$$, this is computed over all layers,
from the input to the last layer $$L$$, with a weighting $$\lambda_{l}$$ for each layer's contribution loss:

$$\mathcal{L}_u =\sum_{l=0}^{L} \lambda_{l}\|\mathbf{z}^{(l)}-\hat{\mathbf{z}}^{(l)}\|^{2}$$

If the input $$x$$ in a labeled data point $$x \in \mathcal{D_l}$$. Then we can add a supervised loss term
to $$\mathcal{L}_u$$ to obtain the final loss. Note the supervised cross-entropy
loss is computed between the corrupted output $$\tilde{y}$$ and the targets $$t$$:

$$\mathcal{L} = \mathcal{L}_u  + \mathcal{L}_s = \mathcal{L}_u + \log P(\tilde{\mathbf{y}}|t)$$

<figure style="width: 75%" class="align-center">
  <img src="{{ 'images/SSL/ladder_network.png' | absolute_url }}" alt="">
  <figcaption>Fig. 2. An illustration of one forward pass of Ladder Networks, C refers
  to the MSE loss between the activations at various layers.
  (Image source: <a href="https://arxiv.org/abs/1507.02672">Rasmus et al</a>)
  </figcaption>
</figure>

The method can be easily adapted to convolutional neural networks (CNNs)
by replacing the fully-connected layers with
convolution and deconvolution layers for semi-supervised computer vision tasks.
However, the ladder network are quite heavy computationally, approximately tripling
the computation needed for one training iteration. To mitigate this,
the authors propose a variant of ladder networks called **Γ-Model** where
$$\lambda_{l}=0$$ when $$l<L$$. In this case the decoder is omitted and the unsupervised loss
is computed as the MSE between the two outputs $$y$$ and $$\tilde{y}$$.

$$

## Π-model 

The **Π-model** ([Laine et al](https://arxiv.org/abs/1610.02242)) is a simplification of the **Γ-Model** of Ladder Networks,
where the denoising encoder is removed and the same network is used to get the prediction for both corrupted and uncorrupted inputs.
Specifically, **Π-model** takes advantage of the stochastic nature of the prediction function $$f_ \theta$$ in
neural networks due to common regularization techniques, such as data augmentation and dropout that typically don't alter the model's predictions.
For any given input $$x_i$$, the objective is to reduce the distances between two predictions of $$f_ \theta$$ with $$x_i$$ as input in both forward passes.
Concretely, as illustrated in Figure 3, we would like to minimize $$d(z_i, \tilde{z}_i)$$ where we consider one of the two outputs as a target.
Given the stochastic nature of the predictions function (ie., using dropout as noise source),
the two outputs $$f_\theta(x_i) = z_i$$ and $$f_\theta(x) = \tilde{z}_i$$ will be distinct. And the objective is
to obtain consistent predictions for both of them. In case the input $$x$$ is a labeled data point, we also compute the cross-entropy supervised loss using the provided labels $$y_i$$ and the total loss will be:

$$\mathcal{L} = w(t)\ d_{\mathrm{MSE}}(z_i, \tilde{z}_i) + y_i\log(z_i)$$

With $$w(t)$$ as a weighting function, starting from 0 up to a fixed weight $$\lambda$$ (eg., 30) after a given number of epochs (eg., 20% of training time). This way, we avoid using the untrained and random prediction function providing us with unstable predictions at the start of training to extract the training signal from
the unlabeled examples.

<figure style="width: 100%" class="align-center">
  <img src="{{ 'images/SSL/pi_model.png' | absolute_url }}" alt="">
  <figcaption>Fig. 3. Loss computation for <b>Π-model</b>, we compute the MSE between the two outputs for the unsupervised loss, and if the input
  is a labeled example, we add the supervised loss to the weighted unsupervised loss.
  (Image source: <a href="https://arxiv.org/abs/1610.02242">Laine et al</a>)
  </figcaption>
</figure>

## Temporal Ensembling

Π-model can be divided into two stages, we first classify all of training data without updating the weights of the model,
obtaining the predictions $$\tilde{z}_i$$, and in the second stage, we consider the predictions $$\tilde{z}_i$$ as targets for the unsupervised
loss and enforce a consistency of predictions by minimizing the distance between the current outputs $$z_i$$ and the outputs of
the first stage $$\tilde{z}_i$$ under different dropout and augmentations.
The problem with this approach is that the targets $$\tilde{z}_i$$ are based on a single evaluation of the network and can
rapidly change, this instability in the targets can lead to an instability during training and reduces the amount of training 
signal that can be extracted from the unlabeled examples. To solve this, ([Laine et al](https://arxiv.org/abs/1610.02242)) proposed a second
version of Π-model called **Temporal Ensembling** where the targets $$\tilde{z}_i$$ are the aggregation of all the previous predictions.
This way, during training, we only need a single forward pass to get the current predictions $$z_i$$ and the aggregated targets $$\tilde{z}_i$$,
speeding up the training time by approximately 2x. The training process is illustrated in Figure 4.

<figure style="width: 100%" class="align-center">
  <img src="{{ 'images/SSL/temporal_ensembling.png' | absolute_url }}" alt="">
  <figcaption>Fig. 4. Loss computation for <b>Temporal Ensembling</b>, we compute the MSE between the current prediction and 
  the aggregated target for the unsupervised loss, and if the input is a labeled example, we add the supervised loss to the weighted unsupervised loss.
  (Image source: <a href="https://arxiv.org/abs/1610.02242">Laine et al</a>)
  </figcaption>
</figure>

For the targets $$\tilde{z}_i$$, at each training iteration, the current outputs $$z_i$$ are accumulated into the *ensemble outputs* $$\tilde{z}_i$$ 
by an exponentially moving average update:

$$\tilde{z}_i \leftarrow \alpha \tilde{z}_i+(1-\alpha) z_{i}$$

where $$\alpha$$ is a momentum term that controls how far the ensemble reaches into training history. $$\tilde{z}$$ 
can also be seen as the output of an ensemble network $$f$$ from previous training epochs, with the 
recent ones having a greater weight than the distant ones.

At the start of training, temporal ensembling reduces to Π-model since the aggregated targets are very noisy,
to overcome this, similar to the bias correction used in Adam optimizer, the training targets $$\tilde{z}$$ are corrected for the startup bias 
at training step $$t$$ as follows:

$$\tilde{z}_i \leftarrow (\alpha \tilde{z}_i+(1-\alpha) z_{i}) / (1-\alpha^{t})$$

The loss computation in temporal ensembling remains the same as in Π-model, but with two important benefits. First, the training is 
faster since we only need a single forward pass through the network to obtain $$z_{i}$$, while maintaining
an exponential moving average (EMA) of label predictions on each training example, and penalizes predictions that are inconsistent with these targets.
Second, the targets are more stable during training, which yield better results. 
The downside of such method is the large amount of memory needed to keep an aggregate of the predictions for all of the training examples,
which can become quite memory intensive for large datasets and dense tasks (eg, semantic segmentation).

## Mean teachers

In the previous approach, the same model plays a dual role as a *teacher* and a *student*. Given a set of 
unlabeled data, as a teacher, the model generates targets, which are then used by itself as a student for learning using a consistency loss.
These targets may very well be misclassified, and if the weight of the unsupervised loss outweighs that of the supervised loss,
the model is prevented from learning new information, and keeps predicting the same targets, resulting in a form of confirmation bias. To solve
this, the quality of targets must be improved.

The quality of targets can be improved by either: (1) carefully choosing the perturbations instead of x²x²simply injecting
additive or multiplicative noise, or (2) carefully choosing the teacher model responsible for the target, instead of using a replicate of the 
student model.

<!-- [Antti Tarvainen et al](https://arxiv.org/abs/1703.01780) investigate the second approach and propose
to form a better teacher model from the student model without additional training. -->

Π-model and its improved version with Temporal Ensembling provides a better and more stable teacher model by 
maintaining an EMA of the predictions of each example, which is formed by an ensemble of the model’s current version and those earlier
versions that evaluated the same example. This ensembling improves the quality of the predictions, and using them as the
teacher predictions improves results. The new learned information is incorporated into the training a slow pace, since each target is 
updated only once during training. Additionally, the larger and span of the updates. To overcome the limitations of Temporal Ensembling,
[Antti Tarvainen et al](https://arxiv.org/abs/1703.01780) propose averaging the model weights instead of its predictions and call this
method Mean Teacher, illustrated in the Figure 5.

<figure style="width: 100%" class="align-center">
  <img src="{{ 'images/SSL/mean_teacher.png' | absolute_url }}" alt="">
  <figcaption>Fig. 5. The Mean Teacher method. The teacher model, which is an EMA of the student model, is responsible
  of generating the targets for consistency training. The student model is then trained to minimize the supervised loss
  over labeled examples, and the consistency loss over unlabled examples. At each training iteration, both
  models are evaluated with an injected noise (η, η'), and the weights of the teacher model are updated using the current student model
  to incorporate the learned information at a faster pace.
  (Image source: <a href="https://arxiv.org/abs/1703.01780">Antti Tarvainen al</a>)
  </figcaption>
</figure>

A training iteration of Mean Teacher is very similar to previous methods, the main difference is that were the Π-model uses
the same model as a student and a teacher $$\theta^{\prime}=\theta$$ and temporal ensembling approximate a stable teacher $$f_{\theta^{\prime}}$$
as an ensemble function with a weighted average of successive predictions.
Mean Teacher defines the weights $$\theta^{\prime}_t$$ of the teacher model $$f_{\theta^{\prime}}$$
at training step $$t$$ as the EMA of successive student's weights $$\theta$$:

$$\theta_{t}^{\prime}=\alpha \theta_{t-1}^{\prime}+(1-\alpha) \theta_{t}$$

The loss computation in this case is the sum of the supervised and unsupervised loss, where the teacher model is used to obtain the targets
for the unsupervised loss for a given input $$x_i$$:

$$\mathcal{L} = w(t)\ d_{\mathrm{MSE}}(f_{\theta}(x_i), f_{\theta^{\prime}}(x_i)) + y_i\log(f_{\theta}(x_i))$$

## Virtual Adversarial Training






































### References

[1] Chapelle et al. [Semi-supervised learning book](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf). IEEE Transactions on Neural Networks, 2009.

[2] Xiaojin Jerry Zhu. [Semi-supervised learning literature survey](http://www.acad.bg/ebook/ml/MITPress- SemiSupervised Learning.pdf). Technical report, University of Wisconsin-Madison Department of Computer Sciences, 2005.

[3] Rasmus et al. [Semi-supervised learning with ladder networks](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf). NIPS 2015.

[4] Samuli Laine, Timo Aila. [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242). ICLR 2017.

[5] Harri Valpola et al. [From neural PCA to deep unsupervised learning](https://arxiv.org/abs/1411.7783). Advances in Independent Component Analysis and Learning Machines 2015.

[6] Antti Tarvainen, Harri Valpola. [Mean teachers are better role models:Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780). NIPS 2017.

[7] Takeru Miyato et al. [Virtual adversarial training: a regularization method for supervised and semi-supervised learning.](https://arxiv.org/abs/1704.03976). Transactions on Pattern Analysis and Machine Intelligence 2018.