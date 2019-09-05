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
