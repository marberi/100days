# 100days
100 days of AI/ML. The requirement of the challenge is to study or apply AI/ML
each day for 100 days. Previously I have done this twice, as documented in 
[Round 1](https://github.com/marberi/100days/blob/master/first100.md)) and
[Round 2](https://github.com/marberi/100days/blob/master/second100.md)). The
third round started on September 3rd.


# Day 301 [2019-09-02]
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
