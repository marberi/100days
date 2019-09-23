# 100days
100 days of AI/ML. The requirement of the challenge is to study or apply AI/ML
each day for 100 days. Previously I have done this twice, as documented in 
[Round 1](https://github.com/marberi/100days/blob/master/first100.md)) and
[Round 2](https://github.com/marberi/100days/blob/master/second100.md)). The
third round started on September 3rd.


# Day 201 [2019-09-02]
Today I worked on another way of outputting probablity distributions from the neural
network. The approach I have been taking so far is creating a grid in the 1D output
space and predicting the probability in each grid cell. The output is then constrained
using cross-entropy loss. Instead I experimented with using a Gaussian Mixture Model
(GMM) as the last layer in the network. This is not provided by PyTorch, so I needed
to partially implement something myself. Below is an example of an output:

![GMM probability distribution](https://github.com/marberi/100days/blob/master/gmm_for_lines.png)

The important part here is to not automaticall assume this is actually a
probability distribution. It could simply be a continous line peaking somewhere
around the correct answer, but not without a proper probabilistic interpretation.
After quite some back and forth, it looks like it works when fitting straight
lines. I also had a test on some simulated galaxies, comparing to the more
traditional way of outputting probability distributions. So far it seems promising,
but I need to test using real data.

# Day 202 [2019-09-03]
Read some further blog posts about mixture density networks (MDN) and large parts
of the original paper [Bishop 1994](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf)

# Day 203 [2019-09-04]
Experimented more with these types of networks. The Bishop1994 paper had one very
simple example, generating 1000 datapoints with two variables. For certain input
values, two output values are equally likely. There are simply not enough information
to fully give the output. Here the posterior should be multimodel. I have polished
up the [notebook](https://github.com/marberi/100days/blob/master/mdn_bishop.ipynb).
Below is a reproduction of Fig.7 in the paper

![Bishop1994 Fig.7](https://github.com/marberi/100days/blob/master/bishop_repod_fig7.png)

# Day 204 [2019-09-05]
The mixture density network used an exponential to keep the width of the distribution.
Alternatively, one can use 1+elu, which I saw elsewhere. Here elu is the exponential
linear unit. Above zero it equals ReLU (identity function), but has a smooth transition
around zero and approach -1 when going to minus infintity. I tested training the network
with both approaches five times each. The result is shown below.

![exp versus elu](https://github.com/marberi/100days/blob/master/exp_elu.png)

For this case, the elu converge faster. Further, I was reading up on
[Machine learning in astronomy](https://arxiv.org/pdf/1904.07248.pdf)
while waiting.

# Day 205 [2019-09-06]
Implemented a mixture density network in the distance determination pipeline, being
able to compare with the traditional results. Some of the results does not make sense
at all, with the network not training properly.

# Day 206 [2019-09-07]
Today I managed to get it working. In the end, not fully sure what ended up making
the difference. The results starting to make more sense when adding a non-linear
layer (ReLU) directly before the MDN. This network is quite deep and I had skipped
this single ReLU. Moreover, there was a significant performance problem when 
evaluating the MDN on the test set. Outputting the values on a grid was extremely
fast. For some reason, evaluating the MDN was rather slow. This ended up becoming
a serious bottleneck, since earlier I evaluated the test set metrcis after each
epoch. Only doing this every 50th epoch lead to a significant speedup. This allowed
for creating a sufficiently large run and evaluating the performance. The MDN now
give sensible results for a simplified test, which has removed some of the results
of pretraining on simulations.

# Day 206 [2019-09-08]
Modified the part of the distance estimation which was pretraining on simulation to
also be able to use a mdn. I managed to make some different runs. While working, I
did not achieve better results than constructing the posterior distribution by
combining man different output classes.

# Day 207 [2019-09-09]
Wanted to look at how to input errors to the network, but without much success. Read
through [blog post](https://medium.com/ai%C2%B3-theory-practice-business/top-6-errors-novice-machine-learning-engineers-make-e82273d394db)
and some other articles.

# Day 208 [2019-09-10]
Read through the noise2noise paper.

# Day 209 [2019-09-11]
Read the noise2void paper. In the noise2noise paper, one would not need clean examples
to denoise the image, but multiple noisy realization of the same underlying image. This
is not always possible to get. Quite interesting for one of our applications.

# Day 210 [2019-09-12]
Managed to find some references on how to treat uncertainties in the neural network. The
[Lightweight probabilistic deep learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gast_Lightweight_Probabilistic_Deep_CVPR_2018_paper.pdf)
paper is explaining how to let the activations be probabilistic. For other approaches
the weights would be interpret as Gaussians. This approach is supposed to be significantly
faster and also avoid repeated forward passes, which many probabilistic methods rely
on. I also attempted making another implementation of the MDN network.
Evaluating the probability function is now extremely fast, but the training
seems to be affected. Not sure why.

# Day 211 [2019-09-13]
Found out that the amount of dropout used is critical for getting a good performance. For
classification tasks, you can easily use 20-50\% dropout for good results. I ended up
using 2\% in the early layers. More dropout degraded the results.

# Day 212 [2019-09-14]
Watched interview with [Jeremy Howard](https://www.youtube.com/watch?v=J6XcP4JOHmk&t=4840s).

# Day 213 [2019-09-15]
Continued tweaking of the model, experimenting with the results. The results
are looking quite close to the previous ones when considering all objects.
However, it looks like the result is improving compared to the previous
results when only considering the best 50\%.

# Day 214 [2019-09-16]
Worked on writing an application today, which will contain a significant amount
about deep learning. Is deep learning hype? I read found
[this article](https://medium.com/machine-learning-in-practice/deep-learnings-permanent-peak-on-gartner-s-hype-cycle-96157a1736e) and
[Howard interview](https://www.youtube.com/watch?time_continue=1202&v=i76E6tvey_M)
quite constructive. Also read on [transfer learning](https://arxiv.org/pdf/1812.06055.pdf)

Quote from the paper:
"As an example relevant to ICF, researchers at the National Ignition Facility
(NIF) [18] have used transfer learning to classify images of different types of
damage that occur on the optics at NIF. There are not enough labeled optics
images to train a network from scratch, but transfer learning with a network
pre-trained on ImageNet [13] produces models which classify optics damage with
over 98% accuracy."

That is quite a strech of domain.


===============================================

https://www.youtube.com/watch?v=Ui1KbmutX0k

https://slideslive.com/38917690/multitask-learning-in-the-wilderness
=======================

Friday:
Looked at panel debate about probabililistic networks.

===
Saturday:
https://papers.nips.cc/paper/1624-neural-networks-for-density-estimation.pdf

messed around with code for doing this.

==========
Sunday:
Google AI paper on unsupervised learning. And the res2next paper.
