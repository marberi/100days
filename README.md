# 100days
100 days of AI/ML. The requirement of the challenge is to study or apply AI/ML
each day for 100 days. The first 100 days started from January 4th and
successfully ended 27th of April
([log](https://github.com/marberi/100days/blob/master/first100.md)). After a
break, the next 100 day challenge started on May 2nd.

# Day 101 [2019-05-02]
Today I was preparing for a talk on deep learning on Fridays, using at least 10
hours today. This is mostly about making slides and reading up on all kind of
background material. While it is an internal seminar, it is always good to be
prepared. You also learn a lot yourself. The biggest surprise was learning
about [Habana Labs](https://habana.ai/), which is a startup semiconductor
company. They claim having a interence processor handling 15000 img/sec,
compared to 2657/sec on a Resnet50 architecture. Also, by accident I found
[Katakoda](https://www.katacoda.com/), which seems useful.

# Day 102 [2019-05-03]
At least by now I have a complete set of slides. Basically took the full day.
This talk covers history, trends in artificial intelligence (AI), hardware,
software/Infrastructure, details on neural networks, astronomy applications,
outlook and conclusions. It covers a serious amount of material, so I could
even have spent more time on reading background material.

# Day 103 [2019-05-04]
Gave the talk today. It was well received. This night I was looking into how
to predict the corresponding errors to a prediction. Many of the papers was
extremely old, like from 20 years ago. It was not clear if these would be a
good fit for our problem. In the end I found that searching for "neural network
prediction interval" gave lots of better hits.

# Day 104 [2019-05-05]
Continued working on the drone. By now the YOLO integration in the notebook
seems to work better. The picture below is from a live stream. Things are
progressing in the group. One of the next topics is studying in detail the
drone executing commands and reading out picures as the same time.

![Drone group](https://github.com/marberi/100days/blob/master/drone_group.png)

# Day 105 [2019-05-06]
Watch [Bengio interview](https://www.youtube.com/watch?v=azOmzumh0vQ) and some
other interviews.

# Day 106 [2019-05-07]
Experimented with the Drone API. One of the problem we experienced on Saturday
is having to wait for commands to execute. For example, when rotating the drone
around, we rotated 5 degrees, took one picture and then rotated 5 more. Also,
the commands often fails with a timeout. At time those commands actually appears
to be working. Reading more about the API, the commands are either implemented
using a blocking or non-blocking way of sending a command. For example, rotating
will by default block. For example

```
tello.send_command_without_return("cw 10")
```

rotates 10 degrees clockwise. More commands can be found by reading the API
source code.

# Day 107 [2019-05-08]
Continued looking at the error prediction. The 
[High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach](https://arxiv.org/abs/1802.07167) paper
looked promising, but also require us to constrain additional quantity output
from the network. It looks like repeated training will be hard to avoid.

# Day 108 [2019-05-09]
Yesterday I had technical problems with charging the laptop and got nothing
done. When finding out about the issue it was too late to get something
done.

Today I was looking at some code for unsupervised networks. Basically this
resume a project from quite a while ago. I previously wrote a prototype and
a student has been working on this. Using parts of this code I wanted to
double check some details, since parts are not working as expected.

# Day 109 [2019-05-10]
Continued playing around with some estimation using an unsupervised method. On
one of the first tries I get

![Flux and background](https://github.com/marberi/100days/blob/master/flux_bkg.png)

which shows the estimated relative errors for two quantities. These should
both be centered around one. More work is clearly needed.

# Day 110 [2019-05-11]
Continued working on the flux estimation. Dealing with unsupervised learning
and multiple losses are tricky. I start to understand better what is possible
or not.
![Only flux](https://github.com/marberi/100days/blob/master/flux_pred.png)

# Day 111 [2019-05-12]
First continuing on the flux estimation, then going back to something which
I looked at a long time ago. How can a network both output a prediction and
corresponding error? One concern I had was the network potentially preferring
a large as possible error. However, the width of a Gaussian distribution also
enters into the normalization. If also including this normalization term in
the loss, it will balance. Using some simulations of straight lines and a 
linear network, I find that the outputted errors makes some sense. However,
the scale is slight off. I need to check if there is some factor missing etc.
At least is seems quite promising.

![Error prediction](https://github.com/marberi/100days/blob/master/pred_prod.png)

# Day 112 [2019-05-13]
Was looking through some videos on "Introduction to Practical Deep Learning" on
coursera. At some time I signed up (no payment), since it was thought be Intel.
They are not the most central in machine learning, but it would be interesting
to see what they are upto. The course is based on Neon. Not sure if this makes
me happy. At the end I watched an interview with
![Chris Lattner](https://www.youtube.com/watch?v=yCd3CzGSte8&list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4&index=3&t=0s)

# Day 113 [2019-05-14]
At work someone contacted me concerning a problem with a neural network. They
had simulated gravitational waves and detector background. The idea is to use
a 1D convolutional network to detect gravitational waves. This approach was
previously done in ![Huerta](https://arxiv.org/abs/1711.03121). For some reason
they could not get it working at all. Implementing this in PyTorch gave on the
test sample a perfect score after 4 epochs! Never seen something like that 
before.
