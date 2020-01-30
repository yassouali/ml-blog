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
remarkable perfomances on certain supervised learning tasks (e.g. image classification, language Modeling) when trained on
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
be the targets associated with some of the examples. In this case, the data set
\\(X=(x_{i})_{i \in[n]}\\) can be divided into two parts: the points
\\(X_{l}:=(x_{1}, \ldots, x_{l})\\), forwhich labels
\\(Y_{l}:=\left(y_{1}, \dots, y_{l}\right)\\) are provided, and the points
\\(X_{u}:=\left(x_{l+1}, \ldots, x_{l+u}\right)\\), the labels of which are
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
- **Graph Based Algorithms.** A labeld and unbaled data point the nodes of the graph, and the objective is to propagate the labels from the labeled nodes to the unlabled ones,the similar to nodes \\(n_i\\) and \\(n_j\\) are, the more likely it is that they share the same label.
- **Bootstraping.** A trained model on the labeled set can be used to produce additionnal training examples extracted from the unlabled set, the extracted examples can be based on some heuristic. Some examples of Bootstraping based SSL are *Self-training*, *Co-training* and *Multi-View Learning*.

In this post, a after brief introduction to the mentionned methods, we will focus more on consistency Regularization based approches, given that they are the most commonly used methods in deep learning.

## Assumptions:

