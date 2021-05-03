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

# Day 214 [2019-09-17]
Watched [Yann Lecun](https://www.youtube.com/watch?v=SGSOCuByo24) interview.

# Day 215 [2019-09-18]
Read about [transfer learning in astronomy](https://arxiv.org/abs/1812.10403) and some
other papers.

# Day 216 [2019-09-19]
Looked at a [video](https://www.youtube.com/watch?v=Ui1KbmutX0k) and  
[presentation](https://slideslive.com/38917690/multitask-learning-in-the-wilderness)
about transfer learning and multi-task learning.

# Day 217 [2019-09-20]
Looked at panel debate about probabililistic networks.

# Day 218 [2019-09-21]
One idea I explored earlier was using neural networks for density estimation. The
earlier attempts, some semesters ago, was not working very well. Attempting again,
I read through [paper on density estimation](https://papers.nips.cc/paper/1624-neural-networks-for-density-estimation.pdf)
and made an implementing in the notebook.

# Day 219 [2019-09-22]
Read through a [paper](https://arxiv.org/pdf/1811.12359.pdf) from Google AI on
unsupervised learning. It was voted the best paper on ICML2019. The most interesting
part was actually the style. Many papers are focusing on achieving a minor improvement
on some benchmark. Here the authors was giving some proofs and a lot of tests that
a general unsupervised separation was possible. They has a ridiculles number of pages
in the appendix.

Also read the [res2next paper](https://arxiv.org/pdf/1904.01169.pdf)

# Day 220 [2019-09-23]
Continued with the notebook from day 218, attempting to have it working. Did not
function either after attemping different things for an hour. However, late I
realized at least one problem. Hopefully that will solve the issue.

# Day 221 [2019-09-24]
Watched [Regina Barzilay: Deep Learning for Cancer Diagnosis and Treatment](https://www.youtube.com/watch?v=x0-zGdlpTegI)
interview. There was some interesting points, like her thinking about medical
problems and the potential for using deep learning for early cancer diagnosis. 

# Day 222 [2019-09-25]
Continue working on the reweighting, doing some expressions on paper and implementing
them. As shown below, I have no success yet.
![Reweighting failure](https://github.com/marberi/100days/blob/master/dist_reweight.png)

# Day 223 [2019-09-26]
Read through two papers 
([paper 1](https://arxiv.org/pdf/1711.09919.pdf), [paper 2](https://arxiv.org/pdf/1903.03105.pdf))
on using recursive neural networks for denoising gravitational wave observations.

# Day 224 [2019-09-27]
Watch through [Francois Chollet interview](https://www.youtube.com/watch?v=Bo8MY4JpiXE&feature=youtu.be), at least
partially.

# Day 225 [2019-09-28]
Looked at the [Turing Lecture](https://www.youtube.com/watch?v=VsnQf7exv5I) talk.

# Day 226 [2019-09-29]
Next video with [Susskind](https://www.youtube.com/watch?v=s78hvV3QLUE&t=2179s) in the AGI
podcast series. He is a well known persons from the physics community. A bit too many
vidoes in a row by now.

# Day 227 [2019-09-30]
First test of actually using the PAUS spectras. Here I used the simulations to
predict the distance from a subset of the observations. Without noise the results
was better than expected from what I previously has read in the literature. A bit
unecpected.

![Distance test](https://github.com/marberi/100days/blob/master/redshift_test.png)

# Day 228 [2019-10-01]
Read through a [paper](https://arxiv.org/pdf/1705.05620.pdf) which used denoising
autoencoders to unsupervise feature extraction from galaxy data. This is relevant
for what I am doing.

# Day 229 [2019-10-02]
Worked on writing a proposal based on deep learning methods. While doing this
application for a while, today I spent 6 hours on the main project part.

# Day 230 [2019-10-03]
Wanted to look more into multi-task learning. I read an
[overview paper](https://arxiv.org/pdf/1706.05098.pdf) from Sebastian Ruder.

# Day 231 [2019-10-04]
One problem I have looked on many times is using neural networks for speeding
up the calculations of simulations. Basically, you have a large training sample
with noiseless examples and would like to train a network to be able to produce
a billion new examples, conditioned on some parameter. Training a network, it
kind of work, but the error on the output was 2-3% at best. This is not
sufficiently good for our application. Finding relevant literature was not
easy. In the end, I figured out the magic searchword was "interpolation". 

# Day 232 [2019-10-05]
Watched [Gary Marcus: Toward a Hybrid of Deep Learning and Symbolic AI](https://www.youtube.com/watch?v=vNOTDn3D_RI)
interview. It was quite interesting hearing from an insider talking about what
he considered some limitations of current deep learning systems. Abstract
concepts seems to be missing in deep learning networks and it unclear how
to build them in.

# Day 233 [2019-10-06]
Continued looking at the interpolation network. Instead of only outputting the
value, I also returned the error. The hope was to use this to downweight problematic
points. In the image below, the three columns are the training loss, test loss and
the relative error. It did not work very well. I also tried other sources of tweaking.

![Not coverging](https://github.com/marberi/100days/blob/master/not_converging.png)

# Day 234 [2019-10-07]
Last full day of writing a proposal which has a strong machine learning
component. Looked up references on multi-task learning, including [Multi-task
learning book](https://dl.acm.org/citation.cfm?id=262872)

# Day 235 [2019-10-08]
Continuing about obsessing about the interpolating of measurements using neural
networks, looking at [paper] (https://arxiv.org/abs/1906.05661)

# Day 236 [2019-10-09]
Also played around with the interpolation, reading up some papers.

# Day 237 [2019-10-10]
Continuing with trying to predict the fluxes. Both using ELU and trying to
reduce the prediction to using a single band.

![Relative error](https://github.com/marberi/100days/blob/master/uncertain_pred.png)

This was after reading some paper which talked about a smooth non-linear transition
would work better than a simple ReLU. That did not seem to be the case.

# Day 238 [2019-10-11]
Looked into various sources of tranfer learning, including the paper
[A survey on transfer learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf).

# Day 239 [2019-10-12]
Not very effective. I was searching if there was some new trends, looking at
various articles.

# Day 240 [2019-10-13]
Read through quite some papers, being away from the laptop today.

In [Shapely framework](https://arxiv.org/pdf/1910.04214.pdf) the authors introduce
a way to determine if improved performance comes from a change in the algorithms
or the data. Quite technical.

[Foggy scenes](https://arxiv.org/pdf/1910.03997.pdf) tested the improvements
when degrading scenes with artificial fog. This is possible when having a 3D
model and a simple model of the fog, where the transparency is distance
dependent.

[Harware acceleration](https://arxiv.org/pdf/1910.03060.pdf). Not the most
interesting paper. The compared running on CPU or GPUs. The paper did not
give a good impression.

# Day 241 [2019-10-14]
Watched [Watson interview](https://www.youtube.com/watch?v=Whtt2H5_isM) on how
they constructed Watson and beat the best human player in Jeopardy. He had an
interesting perspective on how the project was run. With a difficult task, it
is easy to assume achieving the goal would require inventing something 
completely new. Instead they mostly used existing technology and let different
groupd invent on separate parts. 

# Day 242 [2019-10-15]
I wanted to test a specific transfer learning technique. For this I needed a
simple example. For this I constructed a CNN which could determine the
frequency of a wave. Testing this simple example, it did not work at all. 
Which should not be the case. At the end of 1.5 hours I had removed all
complications, but the network was still not working.

![Sinus recovery](https://github.com/marberi/100days/blob/master/sin_not_recovered.png)

Update: In the end, the problem was located in two matrices in the loss function
being broadcasted differently than expected. Not too easy to detect, since I
directly afterwards did a mean of the results.

# Day 243 [2019-10-16]
Worked on adapting a new set of simulations and then pretrained the network with
these. It was looking good so far.

# Explanation of the missing days.
In the middle of the third round between 200 and 300 days, I started loosing the
interest. Around the same time I worked on finishing up a research paper using
deep learning techniques. Focusing on finishing this paper felt more productive
at the time. This resulted in a very long time where I did not follow this good
habit of working on deep learning each day. One pandemic later, I am finally
back again.

# Day 244 [2021-04-30]
Consider having a distribution and wanting to create a density estimator (like
the KDE). Is it possible todo this with a neural network? Previous attempts that
I had on this failed. By now I tested creating a neural network which gets a
single constant input. The output is given by using a mixture density network.
Below is a plot showing how this fits.

![Model probability](https://github.com/marberi/100days/blob/master/model_prob.jpg)

# Day 245 [2021-05-01]
Another problem with the MDN is when having a multi-variate distribution. One can
in a simple way return multiple independent predictions. The problem is when there are
correlation between the various predictions, which is very often the case. Below is
an example of two correlated Gaussians.

![Multi variate](https://github.com/marberi/100days/blob/master/simple_multi_variate.jpg)

In general there were not a lot of useful literature on the topic. The technical
note
[Training Mixture Density Networks with full covariance matrices](https://arxiv.org/pdf/2003.05739.pdf)
had some useful tips. Some (Cholesky factorization) was along what I though of
doing, but it included also some other ideas (eq.10). Tomorrow I hope to make
an implementation.

# Day 246 [2021-05-02]
Worked on actually making the implementation, fitting the MDN to a 2D Gaussian with
a correlation between the variable. The result


![First try on 2D MDN](https://github.com/marberi/100days/blob/master/ifirst_try_2d.jpg)

shows some more work is needed.

