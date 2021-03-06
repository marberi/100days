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
[Chris Lattner](https://www.youtube.com/watch?v=yCd3CzGSte8&list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4&index=3&t=0s)

# Day 113 [2019-05-14]
At work someone contacted me concerning a problem with a neural network. They
had simulated gravitational waves and detector background. The idea is to use
a 1D convolutional network to detect gravitational waves. This approach was
previously done in [Huerta](https://arxiv.org/abs/1711.03121). For some reason
they could not get it working at all. Implementing this in PyTorch gave on the
test sample a perfect score after 4 epochs! Never seen something like that 
before.


# Day 114 [2019-05-15]
Watched some long interviews on
[Social impact of AI](https://www.youtube.com/watch?v=FYIVX5sFeZY)
[Ray Kurzweil](https://www.youtube.com/watch?v=9Z06rY3uvGY)

# Day 115 [2019-05-16]
Attempting to train a network in an unsupervised way. The results are quite
good, except missing an overall scale of the prediction. This is something
we actually can live with, since the scale needs to be calibrated anyway.
However, having this scale running off is not good. Balancing this with a
second loss without screwing up predictions is tricky. Not success yet, but
I plan keep trying.

# Day 116 [2019-05-17]
In certain subfield of astronomy, the self organized maps (SOM) is a popular
unsupervised algorithm. Each object is mapped into a 2D cell. Because of the
2D structure, it is quite suitable to visualize the final result. One question
is why not using GPU to speed up this training process? Actually, it turned
out to be extensive work on this from quite some years back. For example:
[SOM on GPU](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2012-35.pdf)
I was reading upon what people did earlier. A bit historical, seeing their setup
many (7) years back.

# Day 117 [2019-05-18]
Experimenting with different public SOM codes. Implementing something from
scratch is very tempting for educational purposes. However, it would be very
neat if one package worked out of the box. The
[Somoclu](https://somoclu.readthedocs.io/en/stable/)
was quite complete, with multi-processor CPU support. Getting the CUDA support
working required compiling from sources and properly setting the CUDAHOME
variable. When finally running on the GPU, it was actually slower than on a
multi-core CPU. One could even see the GPU not being fully utilized. A bit
disappointing.

After trying some codes, I found
[som-pytorch](https://github.com/meder411/som-pytorch)
which was a small code implementing SOM using PyTorch. Using it was quite
flaky, with the main code focused on running some test examples. Also, it
only worked on GPU. I hacked in support for CPU. Basically replacing .cuda()
calls with .to(device). For one specific benchmark, the runtime decreased from
0.15 seconds to 0.0027 seconds. This was a speedup of 58.7. Quite impressive.

# Day 118 [2019-05-19]
Implemented my own SOM version and did some benchmarks. The implementation was
based on PyTorch tensors. It was a quite useful exercise in how to rapidly
implement a more custom algorithm for GPUs without handcoding CUDA. Below is
a benchmark of one training epoch on two different machines:

![SOM benchmark](https://github.com/marberi/100days/blob/master/som_benchmark.png)

Here the CPU version runs on multiple cores. The Titan-V is about 58 times faster
than using the CPU on the same host.


# Day 119 [2019-05-20]
Went back looking at the galaxy distances. This is fundamentally a regression
problem. It is currently implemented by the network output representing the
probability at a redshift grid. I attempted to instead outputting the mean and
width of this distribution. While sort of working, it does not work as good as
the old solution and takes longer to converge.

# Day 120 [2019-05-21]
Among other things, read through
[REGULARIZING NEURAL NETWORKS BY PENALIZING CONFIDENT OUTPUT DISTRIBUTIONS](https://arxiv.org/pdf/1701.06548.pdf)
which had a very interesting comparison between label smoothing and another term
that can be used to penalize too confident predictions. Previously I found using
label smoothing improved my result, but I needed to tweak a hyperparameter for
different parts of the sample. Looking if this can work for all galaxies at the
same time.

# Day 121 [2019-05-22]
Working on some ideas of smoothing out the output classes. Below is an example
of a predicted probability distribution. While the predictions peak at sensible
values and the width is fine, this distribution should not be so irregular. I
have one idea of how to improve the situation. The attempt of implementing this
gave some weird results, having the loss becoming nan. In the end I could track
this back to a normalization issue. Hopefully I figure out how to solve this
tomorrow.
![Smoothing](https://github.com/marberi/100days/blob/master/pz_problem.png)

# Day 122 [2019-05-23]
Today I manage to find a way to smooth the labels, using a different smoothing
value for each galaxy. And without having to specify this value. Quite happy
about this progress.
![Label smoothing](https://github.com/marberi/100days/blob/master/sigma68_varying_snr_v2.png)

# Day 123 [2019-05-24]
Experimented with using the individual measurements. Usually, we are making
repeated observations of the same galaxy in the same wavelength. The measurements
are then combined in an optimal way, assuming they have a Gaussian noise. In
addition, there are outlier measurements. These are bad measurments or data
reduction issuees, which are not affected in the errors. When trainings a 
network, I usually combines all redundant measurements into one. However, one
would expect using the individual measurementes would perform better, since it
can filter out bad measurements. Doing some tests on a simple simulation, I did
not find a difference. This simulation might have been way too simple.

# Day 124 [2019-05-25]
Today I spents about five hours working on transforming the input data to the
network. This is not trivial, since among other things, I have 12 million input
measurements and wanted to select the best 5 for each galaxy and optical band.
Which might not be the correct thing afterall, but I got it working. Much of
the time was working with Dask, knowing how you can transform you data. Dask
is an excellent tool, which includes an interface looking much like Pandas. It
works quite well, with some exceptions. In the end, I stopped after finding a
weird issue of the code crashing when it shoulds. I will look into this in the
morning. Below is a happy time, processing the dask dataframes is parallell on
16 cores. Quite fun.

![Dask progress](https://github.com/marberi/100days/blob/master/dask_in_progress.png)


# Day 125 [2019-05-26]
Actually tracking down the issue took me two hours. The error first appeared after
modifying a SQL query used to generate part of the input data. At first, I had
forgotten an additional WHERE clause, ending up giving me the same measurement
with three different methods. When fixing this, a weird error started to appear.
Today I worked on stripping the problem down to a minimal example. It turns out
this error randomly gets triggered for the same input data. In the end, I filed
the following [bug report](https://github.com/dask/dask/issues/4845).

# Day 126 [2019-05-27]
Today I implemented the changes needed for running the networks using individual
measurements. This took quite some time. As feared and expected, this did not 
produce better results on the first try. The results is not terrible, but adding
this additional information degrades the accuracy of the predictions. By now I
look into ways to improve upon these results.

To simplify, each galaxy we consider is measured 5 times with each of the 40
different optical bands. This repetiton is to gain a stronger signal. Attempting
to give the network 200 inputs, clearly did not work. When inputting the data,
the ordering of the 5 measurements in each band should not matter. And the number
possible orderings of 5 numbers is 120. For 40 bands, there are a total 10^83
combinations. It would be useful to feed these to the network.

Generating random permutations is not as simple as it seems. At least if wanting
it very fast. Another small contribution adds up when running hundreds of
epochs with 10000 galaxies in batch sizes of 100. And then you want to train
multiple folds. It is possible, but this is also one out of many possible
modifications. 

Below is a benchmark of different ways of generating random permultations. The
first test generates the permulations with a for-loop and the PyTorch "randperm"
function. Instead, one could also generate a lookup table of all the 120
permutations of five numbers (using itertools.permutations). Then one instead draw
random intergers in [0,190] and find the corresponding permulations with this
table. It is 44 times as fast.

![Generate permutations](https://github.com/marberi/100days/blob/master/gen_permutations.png)
# Day 127 [2019-05-28]
Struggeled with one of the models having way more parameters than it should. This
was problematic when attempting to deploy it. Further, focused on making sure the
model could be deployed on multiple machines, only using one core and staying under
2GB memory usage. That might seem strange, but will allow us to run a massive amount
of parallel job. We have infrastructure here which works in that way.

# Day 128 [2019-05-29]
Experimented with running the pipeline from Dask. This was trickier than expected.
For some reason, the map_partition functionaly is not working in the expected way.
Also, I have some issues about controlling the memory usage.


# Day 129 [2019-05-30]
Prepared a presentation on ML in the morning. We had visiors from a biology group,
which was among other things, interested in using ML. Also watched an interview in the
AGI series from Lex Fridman. Interesting discussion at the end about the problem
of unemployment as a consequence of AI. Nothing conclusive, but Eric Weinstein
was sceptical to socialism (read: basic income), which is often the only really
serious proposal that people have to unemployment.
[AGI video](https://www.youtube.com/watch?v=2wq9x2QcZN0&t=185s)

# Day 130 [2019-05-31]
More annoyance with Dask. Spent several hours trying out different approached which
should have worked to run the neural network prediction in parallel. Some of these
problems looks like actual bugs in Dask itself.  

# Day 131 [2019-06-01]
Prepared the AI Saturday presentation. At home I continued investigating the Dask
problem. It seems the easiest if not bothering to collect the result together at
the end, but let each subprosess write to a different file. Kind of cheating.

# Day 132 [2019-06-02]
Changed the wrapper for running a neural network over lots of images to be able
to use a different architecture. This was surpisingly time consumings. Also fails
for some images, with an error which should not be there.

# Day 133 [2019-06-03]
Read through [Multi-task loss](https://arxiv.org/abs/1705.07115) and partly another
paper on the plane. It was quite interesting, proposing a method for being able to
combine multiple losses. Originally we looked at this paper, because we needed to
estimate errors of our predictions.

# Day 134 [2019-06-04]
Throughout the day, I did some debugging for the bug I reported on day 125. It is
kind of tricky, because there are multiple threads involved. In the end I produced
a fix, which might or might not be accepted. 

Continuing working on the network using individual exposures, I figured out how
to actually do the indexing. What I needed was "torch.gather". Kind of hard without
this functionality. By now I am running over simulations to see if this improves
the results.

# Day 135 [2019-06-05]
Worked on applying the technique to data. For some reason it doen not work properly.

# Day 136 [2019-06-06]
Continuing testin using the individual fluxes. Since it did not work, I was testing
many different ideas. For example how to specify which fluxes to mask, data normalization,
etc.

# Day 137 [2019-06-07]
Instead of the classical approach of simply adding the new impormation, we shoud be
a bit more clever. During my travel today I read some papers for inspitation. Doing that
I tested a new technique for combining multiple repeated observations. Normally one would
weight these based on the associated errors. Here I was both using an autoencoder and
also finding the appropriate weights for the different repeated measurements.

![Reconstruction error](https://github.com/marberi/100days/blob/master/new_line_recon.png)

# Day 138 [2019-06-08]
Worked a few hours on applying the code to read data and writing up the results. With
the observations, unlike simulations, one does not have the ground truth. One result is
looking at how the weights of different repeated measuments change. In the below plot,
the blue line shows the dispersion if basing the statistical error. Another line is
based on what an autoencoder predicts. The autoencoder has larger variations in how
much the weighting should vary, which is what one would expect.

![Chi2 in reconstruction](https://github.com/marberi/100days/blob/master/weight_variations.png)

# Day 139 [2019-06-09]
Watched through [Tensorflow interview](https://www.youtube.com/watch?v=NERNE4UThHU). The
largest surprise was how the interviewer considered the open sourcing of Tensorflow to
be a seminal moment in deep learning. Perhaps I have taken it a bit for granted. I actually
remembering trying to find a deep learning library after it became popular, but before
Tensorflow existed. Kind of a pain.

# Day 140 [2019-06-10]
Continued working on implementing ideas for using the individual exposures, using real
data. Managed to train and it looked very promising.

# Day 141 [2019-06-11]
In the morning I thought this changed had improved the perfomance. I was very happy for
a long time, untill discovering switching up the training and test data loader. Not good.
It is a lengthy code, so this can happen.. at least I figured it out before presenting
the results. In the end I managed to trace down another error in the code. So back to
checking what works tomorrow.

# Day 142 [2019-06-12]
Quite productive day. After continuing testing this method, I gave up the current 
approach. While it had some theoretical benefits [not described here], it would require 
minimizing ~200 parameters per input sample. Such a thing is done when doing a style
transfer. Things got a bit complicated, but more importantly, degraded the results.

Instead, the information of individual exposures can be included throught data
augmentation. When training the network, I leave out some measurements at random.
With this I see a significant improvement in my metric. I still have to check the 
value of test time augmentation. Exciting times. For example, the outlier rate
dropped from 17% to 10%. [Unfortunatly I can not post a figure here, since the
data is private.]


# Day 143 [2019-06-13]
Watched [Interview](https://www.youtube.com/watch?v=q0mokx-iiws) with head of the
AI research at Adobe.

# Day 144 [2019-06-14]
Experimented with test time augmentation. In addition to training time augmentation, I
was experimenting with systematically removing some of the repeated measurements. In the
plot below is the estimated distance for a single galaxy. The x-axis is the distance and
a vertical line gives the true distance. The black line shows the probability assigned
to finding a galaxy at a given distance, using the standard network. The peak is too far
from the truth (red line), so this is considered an outlier. 

The set of coloured lines systematically drop some measurement. As one can see, a few of
the lines then locks onto the correct solution. This is hinting towards these measurements
actually being outliers. Looking at the data, exactly these examples are quite far off. Now
I need to find a method to handle them. This is tricker than simply remove them, since the
original data is actually quite noisy.

![Test time augmentation](https://github.com/marberi/100days/blob/master/pz_test_augmentation.png)

# Day 145 [2019-06-15]
Wrote 1.5 pages in the corresponding paper. This is an early draft, but there is always good
to have summarized some progress.

# Day 146 [2019-06-16]
**The posts for the next days is slightly shorter than usual. This mainly comes from the
laptop screen stopped working. So I could not properly type the logs. Also, there were some
days where I was only able to watch videos and read papers from my phone.**

Experimented with test time augmentation, randomly leaving out data. This is similar to what
some people do with images, testing how flips etc. changes the results. Trying many different
configurations, this did not seem to help

# Day 147 [2019-06-17]
Watched through [Rosalind Picard](https://www.youtube.com/watch?v=kq0VO1FqE6I) video. I often
end up watching this video series when the videos come out. A interesting segment is where
they talk about detecting emotions based on visual signals which are not noticable for humans.
For example slight redding of the skin color.

# Day 148 [2019-06-18]
Experimented with using test time augmentation, but in a different way. Instead of using the
test time augmentation to improve the results, I attempted using the repeated measurements
with slightly different data to have an idea of the uncertainty in the measurements. This gave
some very interesting results. **I wanted to continue this at home, but here the computer
broke down**

# Day 149 [2019-06-19]
Watched [Yann Lecun](https://www.youtube.com/watch?v=U2mhZ9E8Fk8) talk and some other videos.
Not quite sure which right now.

# Day 150 [2019-06-20]
Was reading up on multi-task learning. The [MTL blog post](http://ruder.io/multi-task/) was
among the more interesting. My need is slighty different than others, since I want to optimally
combine training with simulations and data.

# Day 151 [2019-06-21]
Worked on finding different ways to characterize the quality provided by the network. In addition
to having good measurements, we are also interested in finding ways to select the 50% galaxies
with the best performance. I tested using some more classical methods on the network output. 
Applying the selection gives a better results, but I had hoped for a larger improvement. I started
reading up on how to use networks themself to rank results. The 
[AirBnB ranking](https://arxiv.org/pdf/1810.09591.pdf) paper had some quite interesting remarks,
like "Out of sheer habit, we started our worst model by initializing all
weights and embeddings to zero, only to discover that is the worst way to start
training a neural network".

# Day 152 [2019-06-22]
Read through [Youtube ranking](https://ai.google/research/pubs/pub45530) and some other resources.
While not exactly being what I was looking for, it was an interesting read.

# Day 153 [2019-06-23]
What I am interested in is not a general ranking, but a selectiong based on the probabilities
associated with the output. Reading the paper
[Probability calibration](https://arxiv.org/pdf/1706.04599.pdf)
it has some interesting results on how techniques used for training a neural network can improve
the accuracy, but destroy the interpretation as probabilities. They discuss a model with a free
parameter which can be used to solve this issue.

# Day 154 [2019-06-24]
Continued working on the same topic, watching a video of their actual paper presentation. Also
thinking of how to best implement and test this technique. They use the validation set to
optimize a single parameter, which adjust the logits in the different classes. It is not 
completely clear this could not be done with the training set.

# Day 155 [2019-06-25]
Worked on various topic throughout the day. I implemented the scaling they suggested in the
Guo et.al. paper. Instead of rigourously using a validation set to determine the correct
parameter, I simply tried a grid of different temperature values. While it can help to
adjust the probability distribution, it does not look like magic.

# Day 156 [2019-06-26]
Used four hours refactoring the code for being able to experiment with another idea for
ranking the galaxies. It use a separe network for providing the relative
ranking. By now it is implemented and I have started to run the first tests. So far
the ranking is essentially random.

# Day 157 [2019-06-27]
Figured out one bug, then continued running many experiments to optimize the result. It
works, but the method is less effective than I need.

# Day 157 [2019-06-28]
Instead of using a rank based on the output, I worked on training a network which was
trained on an intermediated state of the network. Works in a weak sense. Not that
useful in the end. Also attended some talks at UPC, which was part of their deep learning
summer school.

# Day 158 [2019-06-29]
Watch [AlphaGo video](https://www.youtube.com/watch?v=3N9phq_yZP0) talk on AlphaGO. Quite
recommended. 

# Day 159 [2019-06-30]
From the previous talk, I found out there is an AlphaGo documentary. This one was also
nicely made. While knowing much about the facts from before, hearning the comments from
the European GO master, developers etc. gives another depth. It was also interesting to
hear about a specific move, which surprised all experts and which AlphaGo itself considered
only 1/10000 human players would make.
[AlphaGo documentary](https://www.youtube.com/watch?v=jGyCsVhtW0M&t=4659s)

# Day 160 [2019-07-01]
Worked some hours revisiting simpler methods for selecting high quality galaxies based
on the output probabilities. Experimenting with different thresholds, I managed to get
something that works better than on the first pass. This shows that even if both methods
output probability distributions, their exact behaviour on the tail might be different.
This can possible partly be related to the Guo el.al paper mentioned at day 155.

# Day 161 [2019-07-02]
Read through a paper on deep learning and creativity. Not really worth positing the 
link.

# Day 162 [2019-07-03]
Experimented with training multiple networks. Untill now I did a 80-20 split, training
5 independent times to get a full catalogue. Doing this repeatedly with different splits
turns out to give a significant improvement. In my case, I have been training a classifier,
where the classes represent points on a continuum. When finding the best predictions, I
simply take the median value.

# Day 163 [2019-07-04]
Over the night, I trained 500 different networks. Well, actually 1500, since my setup
consist of 3 different networks. In this way I can use a 5-fold (80-20) split and get
100 different predictions. The benefit saturate after a while. I was also working on
stacking the different outputs. That kind of worked, but did not improve the predictions.

# Day 164 [2019-07-05]
Started a new deep learning project. By now I wanted to use deep learning to calibrate
the measurements. Used some hours to download and look at the data.

# Day 165 [2019-07-06]
Continued looking at the data. Specifically, I am interested in multiple measurement
of the same stars. With this we can infer something about the observation conditions.
Below is the picture of a graph, where the nodes are individual measurements and links
indicate the measurements are for the same star. NetworksX is damn slow. I need to 
explore this more.

![Connections](https://github.com/marberi/100days/blob/master/connections.png)

# Day 166 [2019-07-07]
Experimented with the implication of training on simulations with different variations
of learning rates. Basically wanted to see how sensitive the current configuration is
to the exact realization. I have found this earlier to be a bit more sensitive than
what I would expect.

# Day 167 [2019-07-08]

Watched the first hour of an interview with [Jeff Hawkins](https://www.youtube.com/watch?v=-EVqrDlAqYo)
on the AGI podcast series. It was two hours, so I continue tomorrow.

# Day 168 [2019-07-09]
Finished watching the interview. Recommended. It was talking about how the
neocortex, one of the younger parts of the brain, works. This researcher
claims that his team had some fundamental breakthrough in understanding how
it works over the last years. There was talk about how the brain did not
only create one model of what it observed, but multiple models, which it
then voted over. He also discussed how the neocortex looked quite similar
for different areas and how they all seemed to map the information onto a
referene system.

The book which he has published and interviews are older, so I don't know if 
they are out of date.

# Day 169 [2019-07-10]

Experiemented with writing a neural network to mimic the calculations done
in some simulations. I had 136k galaxies simulated using some recipe. This
way of simulating galaxies is fast, but end up becoming time consuming if
needing to simulate billions of galaxies. I tested writing a neural network
for doing this. In the best case, I ended up having a 2% error. This shows
the idea works, but is not sufficiently good for replacing what we currently
have. For this we would need 20x better errors. Possible this would end up
working at a later stage. Getting extremely precise regression with neural
networks is harder than it sounds.

# Day 170 [2019-07-11]
Watched some lectures on transfer learning. I watched this for some project I
wanted to work on.

# Day 171 [2019-07-12]
Continued reading up on transfer learning. Today I was taking a flight, so
it was a good time for reading. Among others, I looked at
[Facial landmark detection through multi-task learning](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf)
and
[Regularized mult-task learning](http://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/evgeniou+pontil-2004.pdf).
The highlight was the
[Simgan](https://zpascal.net/cvpr2017/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf)
paper talking about how to transform simulations to become closer to data. In
this way they achieved a significant improvement when later training a 
regressor for eye positions. This is along the direction of what I could need
for one project. Interestingly enough, I was thinking of this, referring to the
[CycleGAN paper](https://arxiv.org/abs/1703.10593). It turns out the CycleGAN
paper is actually older, referring the SimGAN paper.


# Day 172 [2019-07-13]
Looked at [The Long-term of AI & Temporal-Difference Learning](https://www.youtube.com/watch?v=EeMCEQa85tw) talk.

# Day 173 [2019-07-14]
Experimented with programming a WGAN. It is not my first time programming a
GAN. The previous times I was using it with pictures, mosly cases where others
previously have succeeded. By now I was looking at generating straight lines
with some noise. This is closer to the galaxy fluxes, which I hopefully can
generate later. A critical step was normalizing the data. For some reason, it
did not work well with batchnorm.

The simulations are generated using:

```
N = 10000
SNR = 35

x = np.linspace(0., 100)
A = np.random.random(N)
B = np.random.random(N)
flux_true = A[:,None]*x[None,:] + B[:,None]

err = flux_true / SNR
flux = flux_true + err*np.random.normal(size=err.shape)
flux = torch.Tensor(flux)
```

and when training a generator and discrimination, I find that:

![Generated amplitude](https://github.com/marberi/100days/blob/master/ampl_recovery_v2.png)

While the generator is generating examples which looks drawn from the distribution, the
distribution looks different, generating less of the extreme examples.


# Day 174 [2019-07-14]
Watched interview with [Kai-Fu Lee](https://www.youtube.com/watch?v=cQ48rP_Rs4g) on Chinese and American
artificial intelligence companies. Quite interesting to hear the Chinese perspective. Followed up with
another interview.

# Day 175 [2019-07-15]
Continued playing around with the simple GAN simulations. One of the problematic part can be seen in
the figure below.

![GAN loss](https://github.com/marberi/100days/blob/master/gan_loss_v2.png)

I read through [Which Training Methods for GANs do actually Converge?](https://arxiv.org/pdf/1801.04406.pdf)
searching for a solution. Implementing a combination (ish) of the R1 or R2 regularization terms (eq. 9 and 10),
I find a strong regulizing effect. More results follow tomorrow.

# Day 176 [2019-07-16]
Watched through lesson 12 on GANs in the FastAI course from last year. I had seen it before, but it
was a nice reminder on some of the concepts.

# Day 177 [2019-07-17]
Continued experimenting with the previous GAN simulations, testing the sensitivity to various hyperparameters.

# Day 178 [2019-07-18]
Watched on GAN lecture. Not exactly remember which one by now.

# Day 179 [2019-07-19]
Read through various papers on my flight, focusing on GANs and the adaptation of simultions. One of
the more readable ones was
[CycleGAN for sim2real Domain Adaptation](https://web.stanford.edu/~jaustinb/papers/CS236.pdf).


# Day 180 [2019-07-20]
Worked on preparing my poster for an upcoming workshop on artificial intelligence
in astronomy.

# Day 181 [2019-07-21]
Rerunning the predictions of the figure and finished up the poster.
![Deepz poster](https://github.com/marberi/100days/blob/master/deepz_poster.png)


# Day 182 [2019-07-22]
First day of the workshop. A few half-ways interesting talks. There was one
group from Dubai(?) using drones to look for metorites in the desert. Not very
interesting from a technical viewpoint, but interesting because of the Drone
project in last AI6 edition in Barcelona. I hope to catch her later. Also had
some intersting talks about searching for exoplanets using deep learning in
direct imaging.

# Day 183 [2019-07-23]
Second day of the workshop. There was an intersting talk in the morning on
unsupervised learning. They had been clustering together different objects.
A neat appliation was the search for multiple lensed quasars. A quasar is a
galaxy with a supermassive black hole, often appearing more like a star in
optical images. If being behind a dense dark matter region, the quasar can
appear multiple times, from gravitational lensing bending space. Only 40
of those systems was previously known. Through first using unsupervised
learning, they associated these objects with some classes. This was useful
to know which objects to follow up. In this way they manage to find multiple
systems.

# Day 184 [2019-07-24]
Third day of the workshop. The most interesting talks was about the use of
artificial intelligence for adaptive optics. Light arriving to earth is
passing through the atmosphere, being distorded by the atmosphere. Some
telescopes correct for this using a deformable mirror. Part of the incoming
light is sent to a wavefront sensor, which use this information to correct
using movements of the deformable mirror. This has existed with more 
classical algorithms. The talks was going into detail of how AI algorithms
could improve these corrections.

Today the student was also presenting the work we have been doing on
estimating the background in PAUS images.

# Day 185 [2019-07-25]
Fourth day of the workshop. The biggest surprised was one 3.5h tutorial in the
afternoon on "information field theory". There was also a talk on the same
topic. One advantage of some machine learning technique, is not having to
explicitly model the system. However, injecting information on how you
expect different parts to work is difficult. They was teaching a framework
that relied on building up a model using different steps. It was then
minimizing the coefficients, using backpropagation. Some of the examples
was quite impressive. I think it would have been interesting to see how
this method worked together with non-linear mappings implemented with a
neural network.

# Day 186 [2019-07-26]
Last day of the workshop. The talk I liked the most was one trying to account
for baryons in dark matter simulations. In cosmology one method for measuring
properties of dark matter and dark energy is to correlate the shape of
galaxies. If being intrinsically random, the correlation is caused by
weak gravitational lensing, slightly distorting the images. This will allow
researchers to measure the dark matter at scales corresponding to small
separations between galaxies.

When making this measurements, one need to compare with theoretical models.
These are often done running large n-body simulations, containing trillions
of dark matter particles. However, at small scales, these predictions will
also be affected by baryons. Like from super novaes. Including these in the
large volume simulations is currently not possible. They were discussing a
way of adding this effect to the large simulations afterwards.

# Day 187 [2019-07-27]
A bit down to earth, I worked on running the inference of our model over the
full dataset. Previously I was running this on the CPU. This is because half
of the pipeline is a classical algorithm only running on a CPU and for large
scale deployment we probably end up using CPUs in the beginning. Mostly to
satisfy some conservative engineers. Today I was testing running a single
job on the GPU. This is fine, since we already had the predictions for the
classical algorithm. It went quite fast, but require transferring 790GB over
the network.

# Day 188 [2019-07-28]
Worked on correcting a paper we are writing on background subtracting using
CNNs. It took me around 4 hours.

# Day 189 [2019-07-29]
The method for the background prediction used an embedding to encode the 
information. I experimented with trying to understand the different 
embeddings. One approach used the embedding as the linear combination of
some unknown basis images. These was then constrained comparing with some
average sky-background images in all bands. This resulted in the images
below. Not quite sure about the interpretation yet.

![Embedding information](https://github.com/marberi/100days/blob/master/embedding_interpretation.png)

# Day 190 [2019-07-30]
Among other things, I watched 
[Self-Driving Cars at Aurora, Google, CMU, and DARPA | Artificial Intelligence Podcast](https://www.youtube.com/watch?v=Tj6NOfdfa4o&t=404s)

# Day 191 [2019-07-31]
Experimented with denoising of images. For another project, using an auto-encoder
would be a good solution. I have coded some denoising auto-encoders before. To
remember how this was done, I wrote up some simple simulations looking a bit
like a 1D version of the data and a linear auto-encoder. The result is shown
below. This network was slower to train and did not give as good results as
I expected. In the end, this could have been because I forgot to add some noise
to the input. Well, I will experiment with this again later.

![Denoising](https://github.com/marberi/100days/blob/master/denoise_v3.png)

# Day 192 [2019-08-01]
Read through some papers on denoising networks. In particular [Beyond a Gaussian denoiser]
(https://arxiv.org/pdf/1608.03981.pdf) was interesting. They were describing how training
to predict the residuals was simpler than attempting to predict the cleaned image. This is
interesting because it makes little differene to how there training is actually done. Further,
I looked into image deblurring, including:
[Deep image deblurring](https://papers.nips.cc/paper/5485-deep-convolutional-neural-network-for-image-deconvolution.pdf).
This give me some ideas for applications, but the problem is assumtions which needs to be
slightly different. One would therefore actually have to try in practice to see how well
it performs.


# Day 193 [2019-08-02]
Worked on a gravitational wave project. Here the problem was connected with understanding
if some information from the training set was leaking into the test set. I did some tests
to see how different they were looking.

# Day 194 [2019-08-03]
Continued on the topic of denoising, applying this to 2D simulations. Finally ended up getting
the results below, which was good enough for a quick proof of consept. Now I sent the result
and asked to start a data transfer of real images.

![Atoms 2D](https://github.com/marberi/100days/blob/master/atoms_2d_v2.png)

# Day 195 [2019-08-04]
Worked on trying to apply deblurring to astronomical images. The first results looked
interesting.

# Day 196 [2019-08-05]
Continued on the deblurring, now focusing on varying the size and the amplitude of the
different signals. Here we assume that stars intrinsically are point sources, if it was
not for the blurring of the atmosphere. The network learns to move all flux into a sharp
peak. Unfortunately, this method need to learn using stars, then apply the method to galaxies,
which has an intrinsic width. Since the network never have seen a galaxy during the training,
it tries to put all flux back into a single pixel. This was something I feared and expected
would happen. At least I have some idea of how to address this problem.

![Deblurring galaxies](https://github.com/marberi/100days/blob/master/deblurring_gal_bias.png)

# Day 197 [2019-08-06]
Worked on ideas related to [Cyclegan](https://arxiv.org/pdf/1807.00374.pdf) which is a GAN
transforming between two domains. I thought this could be used to detect some offset between
two samples of galaxies, which should be the same, except some multiplicative factor multiplied
with the observations. Turns out, it kind of. But now sufficiently to understand what the
offset is to percent level. Well, does not destroy the project, but is good to keep in mind.
I feel like I start to get an intuition of where things might fail. Or, these things fails
often and I have a paranoid nature.

# Day 198 [2019-08-07]
Continued with the CycleGAN, following the ACAL paper [for some reason arxiv is down, but
you will find it]. In addition to using the cycle requirement, it also introduced a loss
related to solving a task. The transfer between domains should not only look good in the
cycle, but still be able to map to the correct label. This matches very well with what I
try to achieve. In the end I experimented with a subset of this network. Loooks promising.
Also wrote significant amount of text for the paper.

# Day 199 [2019-08-08]
Worked on extending the setup to include the two transformations and the two descriminators.
I played around training the network, varying losses, training for longer and so on. While
the discribution was shifting to look more realistic, the transformation from simulation
to data did not look fully believable. However, our data is very noisy, so things are
not conclusive.

# Day 200 [2019-08-09]
Watched some problems about training GANs. The first was 
[GANs in the wild](https://www.youtube.com/watch?v=Qc1F3-Rblbw), which was giving some
different tips on how to train the GANs. This included using the labels under points
12. Among other videos, I found one [talk](https://www.youtube.com/watch?v=vDW8qvsBtmQ)
which was for the SimGAN paper. For some reason, I did not find this one earlier.

Then finally the 100 days is over. Too tired to write something sensible by now, but
it was definitively worth the effort.
