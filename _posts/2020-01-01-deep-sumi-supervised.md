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
creating such large datasets requires a considerable amount of ressources, time and effort. Such resources may not be available in a lot of practical settings,
which limits the adoption and application of many deep learning methods.


In search for more data efficient deep learning methods to overcome the need for large
annotated datasets Semi-supervised learning (SSL). Recently, we're seeing a lot
of research interest in developping novel SSL as a possible alternative,
by developping novel methods and adopting existing one for a
deep learning setting
This post discusses Semi-supervised learning (SSL) in a deep learning setting, and goes through some of SSL main methods.

# What is Semi-supervised learning?

> Semi-supervised learning (SSL) is halfway between supervised and unsupervised learning.
In addition to unlabeled data, the algorithm is provided with some supervision
information – but not necessarily for all examples. Often, this information will
be the targets associated with some of the examples. In this case, the data set
$$X=(x_{i})_{i \in[n]}$$ can be divided into two parts: the points
$$X_{l}:=(x_{1}, \ldots, x_{l})$$, forwhich labels
$$Y_{l}:=\left(y_{1}, \dots, y_{l}\right)$$ are provided, and the points
$$X_{u}:=\left(x_{l+1}, \ldots, x_{l+u}\right)$$, the labels of which are
not known.
> 
> <footer><strong>Chapelle et al.</strong> &mdash;
> <a href="http://www.acad.bg/ebook/ml/MITPress-%20SemiSupervised%20Learning.pdf">Semi-Supervised Learning Book</a> </footer>


As stated in the definition above, in SSL, we are provided with a dataset containin both
labeled and unlabeled examples. The portion of labeld examples is usually quite small
compared to the unlabeld example (e.g. 1 to 10% of the total number of examples). So with
dataset $$\mathcal{D}$$ containing a labeld set $$\mathcal{D_l}$$ and an unlaledled set
$$\mathcal{D_{ul}}$$. The objective, or rather hope, is to laverage the unlabeled
examples to train a model and obtain better performance than what can be obtained using only the
labeled portion. And hopefully, get closer to the desired optimal perfomance, in which
all of the dataset $$\mathcal{D}$$ is labeled.

More formally, the goal of SSL is to
leverage the unlabeld data $$\mathcal{D_{ul}}$$ to produce a prediction function
$$f_{\theta}(x)$$, with trainable parameters $$\theta$$, that is more accurate than what would have been obtained by only using the labeled data $$\mathcal{D_l}$$.
For instance, $$\mathcal{D_{ul}}$$ might provide us withadditionnal infomation about the structure of the data distribution $$p(x)$$ to better estimate the decision boudary
between the different classes. As shown in Fig. 1 bellow, when using unlabled data with a SSL approache, we can extract additionnal information about the structure of
the data points and reduce the ambiguity between the two classes.  


<figure style="width: 70%" class="align-center">
  <img src="{{ 'images/SSL/data_manifold.png' | absolute_url }}" alt="">
  <figcaption>Fig. 1. The decision boundaries obtained on two moon dataset, with a supervised and diffrent SSL approaches, using 6 labeled examples, 3 for each class and the rest of the points as unlabeld data. (Image source: <a href="https://arxiv.org/abs/1804.09170">Oliver et al.</a>)
.</figcaption>
</figure> 

Semi-supervised learning first appeared in the form of self-training, where a model is  first trained on labeled data and then, iteratively, at each training iteration, a portion of the unlabeled data is annotated using the trained model and added to the training set for the next iteration. SSL really took of in 1970s after its sucess
with iterative algorithm such as the [expectation-maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) algorithm using labeled and unlabeled
data to maximize the likelihood of the model ([SSL books](http://www.acad.bg/ebook/ml/MITPress-%20SemiSupervised%20Learning.pdf) contains more details about this subject).

