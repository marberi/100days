# 100days
100 days of AI/ML. Study or apply AI/ML for at least 1 hour each of the next
100 days, starting at January 19. 

# Day 1 [2019-01-19]

Started the course on computational linear algebra
[https://github.com/fastai/numerical-linear-algebra/]. I finished part zero and
got halfways into the first.

# Day 2 [2019-01-20]
Finished week 1 of the course. While most was known, they had a nice example
on calculalation stability. Also an interesting example from a linked Halid
talk on how to speedup a naive convolution in C++ with a factor of 11x. It
got me thinking on my own code.

# Day 3 [2019-01-21]
Half the way into video 2. Nothing terrible exciting, but learned that scikit
learn had functionality for fetching newsgroups and creating the vectors.

# Day 4 [2019-01-22]
Finished video 2. Experimented around with the PyTorch implementation of
non-negative matrix factorization. I was surprised how quickly result
changed from being roughly random to make sense. Also found out that
PyTorch implements @ for matrix multiplication. For example A @ B is
the same as A.mm(B). In Numpy you can write A @ B instead of np.dot(A, B).

# Day 5 [2019-01-23]
Almost finished video 3. The most interesting part was when using a random
algorithm for decomposing a matrix. I hope to implement this tomorrow.

# Day 6 [2019-01-24]
Ended up reading through https://arxiv.org/pdf/1211.7102.pdf. Among other
things, the discussed removing singular values in the SVD decomposition to
denoise the image. Here they gave some examples and also suggested using
the Frobenius norm as a way of estimating how many componets to remove
instead of just dropping a fixed number of components.

# Day 7 [2019-01-25]
Finished video 3, homework 1 and started on video 4. A bit slow progression
in the video. At the end Rachel showed how to use SVD for background removal
in videos. Looking forward for exploring that more..

# Day 8 [2019-01-26]
Finished video 4 and half of video 5. The examples was based on a dataset
which no longer is available. Instead I ended up searching for a different
set of data. The set: http://www.svcl.ucsd.edu/projects/background_subtraction/JPEGS.tar.gz
included multiple videos, stored as a series of jpeg files. The SVD algorithm
was suprisingly effective on separating the foreground and background of
an image. Note that this require a steady camera. When attempting to use this
on a video of a plane landing, it considered the plan background and the
airport as moving. Below is SVD used on people walking in the park.

![SVD separation](https://github.com/marberi/100days/blob/master/walking.png)


# Day 9 [2019-01-27]
Finished video 5 and 6, plus doing exercise set 2. Looking more forward for
tomorrow when they will apply the techniques to CT scans.

# Day 10 [2019-01-28]
In the middle of video 7. They are still discussing the CT simulations..

# Day 11 [2019-01-29]
Finished video 7 and 8. It was interesting how much better the CT linear
regression example looked when using a L1 norm.

# Day 12 [2019-01-30]
Half the way into video 9. By now learning about page-rank.

# Day 13 [2019-01-31]
Learned one nice trick on how to estimate the n-th fibonacci number. If
computing it the straigh forward way using recursion, the runtime increase
by 2^b. In this case there is a closed form solution, meaning the answer
is given by a formula. As a result, the answer is almost instantaneous.

![fibonachi](https://github.com/marberi/100days/blob/master/fibonacci.png)

Except this, I continued with downloading the DBpedia dataset and watching
the video.

# Day 14 [2019-02-01]
Finished video 9 and half of video 10. Nothing too new, but I learned what
algorithm is hiding behind some names, like the Hausholder algorithm.

# Day 15 [2019-02-02]
Finished the course on computational linear algebra today. I could have spent
some more time experimenting with the last notebook, but I instead started
going back to the deep learning. Reading a blog post on the 2019 edition on the
fast.ai course introduction course, it seems like they have made significant
changes. Also the new version of the library and notebooks looks better than
the previous versions. Earlier their library had substandard code quality and
messy interfaces. Hopefully ths version is better. It looks more like a designed
library and not something slapped together to support his classes. Probably I
will go through the course again, spending more time reading up on background
material.


# Day 16 [2019-02-03]
![Learning rate finder](https://github.com/marberi/100days/blob/master/fastai_week1_v2.png)

Going through this years fastai lessions definitively seems to be worth it. The
plot shows tweaking a Resnet50 to separate between 37 breeds of cats and dogs.
Previous editions only separated between cats and dogs, but this is now 
considered too easy. The model shown has a 4.9% percent error rate or 95%
accuracy. Their paper achieved 63.5% and 55.7% accuracy for cats and dogs 
respectively, without adding an additional segmentation trick. This is 
quite a large improvement over a few years.

# Day 17 [2019-02-04]
Playing around with the whale classification competition today, making my first
submission. Nothing great [0.284] yet, but a first submission to see that
things was working. I also worked on the fast.ai excersice of creating a
classifier using a custom dataset. For this I downloaded portraits of hipsters
and men (excluding hipsters). This is not the simplest classification task,
given that hipsters are men with a certain style. As seen in the confusion
matrix below, the error is about 7%. By visual inspection, several of the
misclassified images had multiple people and even one woman.

![Hipster finder](https://github.com/marberi/100days/blob/master/hipster_man.png)

# Day 18 [2019-02-05]
At work I resumed a project of trying to use a neural network to determine
galaxy distances. Applying machine learning for this task is quite standard.
However, the data I work with combines more observations with a higher
wavelength resolution. Extracting the high precision distances can be
tricky with a low number of training sets. At home I continued watching
fastai lesson 2. As the picture below show, they include functions which
are slightly different from PyTorch. Not extremely excited about these
hacks.

![Fastai tensor](https://github.com/marberi/100days/blob/master/fastai_tensor.png)

# Day 19 [2019-02-06]
Finished lesson 2 and started on lesson 3. The simple exercise on gradient
decent for fitting the parameters (slope, intercept) of a line was a nice
example. In the/one Pytorch tutorial they started from Numpy and used time
to explain how the autograd functionality saved you from calculating the
gradients by hand.

# Day 20 [2019-02-07]
![Satellite photo](https://github.com/marberi/100days/blob/master/sat_photo.png)

Workin on the multi-label classification part in lesson 3. The largest part of
today was spent dealing with technical issues. I tried downloading and
unzipping the data on a remote server, which turned out to not hae the
compressiong library install. Finding a workaround took some time. Finally
I at least got the images in the end.

# Day 21 [2019-02-08]
Working on tweaking the network accuracy in the (finished) planet competition. 
Training resnet50 takes a bit of time. Not terrible long, but it gives me
some time to wait.

# Day 22 [2019-02-09]
Quite a long days. I played around with training the planet dataset. At the
end of the day, I returned to look at the whale identification dataset. With
some improvements my score went from 0.284 to 0.456. Still not terrible good,
but I gain some experience with this dataset.


# Day 23 [2019-02-10]
Continued fastai lesson 3, looking at segmentation of pictures coming from
dashcams. On the left is the ground truth and the right is the prediction.
While the number of correct pixels here is about 90% [still running tests],
there are systematically some problematic areas. For example the lamp post.  I
also played around with the whales competition. The result when aborting a
longer run midway was not better. I will continue training this
model for a while longer.

![Dashcam segmentation](https://github.com/marberi/100days/blob/master/dashcam.png)

# Day 24 [2019-02-11]
Looking at the text classification with IMDB. For the previous versons of
fastai I skipped NLP, since my work never includes text. Running this task, I
was suprised on how long the training actually took. Instead of using google
colab, I ended up using the local Titan-V. Then it could run in reasonable
time. Below is a test of the classifier separating into positive and
negative reviews. Here the accuracy is 84.2% compared to 94.4% obtained
in the lesson notebooks. This both come from neither unfreezing the 
layers when training the encoder nor classifier.

![IMDB](https://github.com/marberi/100days/blob/master/imdb.png)

# Day 25 [2019-02-12]
Playing around with the prediction of head pose. In some sense, nothing really
special about this application. One input a set of images and predicted the
center of the head. As with many of the other applications, the most challenging
part was inputting the data in a sensible way. Also, as not covered in the 
example, unfreezing and training all layers lead to a significant decrease
of the loss function.

![IMDB](https://github.com/marberi/100days/blob/master/head_pose.png)

# Day 26 [2019-02-13]
Worked on the tabular dataset example in lesson 4. The accuracy was ok
(84%), but not impressive when only wanting to classify into low and 
high income. In the lesson notebook the achieved 82%. Running more epochs,
changing layers etc. did not improve the result. A bit disappointing.
Hopefully they come back with some interesting tweaks.

# Day 27 [2019-02-14]
Most of the day I attempted to fit a large (1 million) simulated galaxy
sample to a neural network. It is unclear if the network is overfitting
or have not properly converged to a state where my metric makes sense.
More work is clearly needed.

# Day 28 [2019-02-15]
Continuing on the work of fitting to the simulations. One problem was,
again, having added a softmax layer when using the cross-entropy loss
from PyTorch. The cross-entropy function in PyTorch includes a softmax
layer and should *not* be included in your network. Also, when training
the network, it seems like the usual normalization of input data does
not work very well. If instead taking the logarithm, then things works
much better. The distribution of one of the features can be seen in the
image. I need to play around to optimize this, since real observations
will have negative measurement coming from statistical fluctuations when
substracting the sky background.

![log_feature](https://github.com/marberi/100days/blob/master/log_features.png)

# Day 29 [2019-02-16]
Last night I was programming a classifier based on images scraped from
the web. The model was trained using the fastai library. When exporting the
network to ONNX, I had problem figuring out which transformations the
fastai library had been using. Other than this, I was watching the PyTorch
developer conference videos from last year.

![valentine](https://github.com/marberi/100days/blob/master/valentine_solutions.png)

# Day 30 [2019-02-17]
Continuing on fastai lesson 4. Today I was working through the collaborative
filtering example in detail.

# Day 31 [2019-02-18]
Finished lesson 4 and got midway into lesson 5. At work I was playing
around with pretraining a neural network using simulations. Finally
it started to give very good results when testing on simulations. After
testing on data I discovered that one parameter range was to small. By
now a new set of simulations are running.


# Day 32 [2019-02-19]
Today I worked on determining galaxy distances with neural networks. Previously
I had problems getting this working, only getting reasonable results when first
pretraining on simulations. With the current setup, which is completely new
since last attempt half a year ago, it ended up giving sensible when *not*
pretraining. Quite unexpected. By now the result is around 1.2\% worse than
a classical method on two important metricts. This is encouraging and gives
me some hope that additional tweaking will give good results.

# Day 33 [2019-02-20]
By now some results are comparable with the classical method!  Continued to
experiment with the distances. The regression problem is implemented as a
classification. This is both done since we need the probability distribution of
the distance and because it gave better results. By now there are classes
without any training samples. This is quite problematic, since these classes
probably are poorly trained.  First attempts on regularizing the layer did not
change the results much. A bit weird.

# Day 34 [2019-02-21]
Watched lesson 5. I had not realized the meaning of weight decay and how it
was implemented. While being an additional term to the loss function, it
can be implemented in the optimizer. Adam has a "weight_decay" parameters
which by default is set to zero. The fastai use 0.01 as a safe default, but
say 0.1 works most times. Looking forward for actually testing this in
practice.

# Day 35 [2019-02-22]
Ended up playing around with MNIST, preparing for the exercise corresponding
to lesson 5. Instead of working with a complicated model, I implemented 
logistic regression. This is a neural net, but without a hidden layer. In
other words, just a single linear transformation. And a softmax layer which
in PyTorch is implicit. Running this resulted in 92.5\% accuracy. Certainly
not state-of-the-art, but not bad either.

# Day 36 [2019-02-23]
First I continued experimenting with MNIST. Using one hidden layer with 100
neurons gave 97.4% accuracy. With a CNN and some tweaks, I managed to pass
99%. Further, I continued working on the galaxy distances. Applying a CNN
on a subset of the data where the input data is regularly space in wavelength
did not make an inprovement. A bit surprising.

# Day 37 [2019-02-24]
Watched most of part 3 of PyToch developer conference from last year. The
biggest surprised was that facebook was having their own open source
implementation of Alphago zero.

# Day 38 [2019-02-25]
Managed to use most of my day on deep learning related material. In the
evening I focused on writing up a pipeline for estimating the galaxy
distances for the full catalogue, using a k-fold method. Not terrible
difficult, but a bit tedious to write. Later I hope to look into PyTorch
ignite, or some framework which can simplify this process. Hopefully the
results looks good enought to be presented tomorrow.

# Day 39 [2019-02-26]
![line fitting](https://github.com/marberi/100days/blob/master/line_fitting.png)

The plan was to experiment with PyTorch ignite. As a test, I created some
noiseless line. A two layer neural network did a very good job on being
able to reconstruct and interception. That is perhaps not very strange,
but I have not checked the formulas yet. The errors became large when
having a small slope. The plot shows the error in the slope (a) as a 2D
plot, where the x and y-axis are the true a and b values, respectively.
The errors plotted have a funky logarithmic scaling to look better. Basically,
higher values are larger values.

# Day 40 [2019-02-27]
Playing around trying transfer learning with simulations. Still seems not
to be working very well.

# Day 41 [2019-02-28]
Read through a paper on "Using convolutional neural networks to predict galaxy
metallicity from three-color images" (arXiv: 1810.12913) which Jan Carbonell
sent. It was a quite nice paper. They even had a discussion and test of how the
network might have indirectly made a prediction through a simpler parameter.
Quite useful to see an example of a CNN paper in our field.

# Day 42 [2019-03-01]
Watched most videos in week 1 of "AI for everyone". Quite basic so far, but
might be more interesting later. It focus more on project success of AI in
companiest than the very technical aspects.

# Day 43 [2019-03-02]
Continuing trying to improve the galaxy distance determination. Previously
using a CNN has not worked very well. While the underlying galaxy spectrum
is a continuoes variable, my input is the measurement in different optical
filters. To get around this problem, I experimented with first adding one
or more linear layer to map back to the spectrum. From there I would be 
using a Resnet like 1D architecture. The paper (arXiv: 1810.03064) had
published this, so it was simple to get started. By now the network fuction
and give some results. Not equally good as the purely linear model, but
I have not experimented too much yet.

# Day 44 [2019-03-03]
The first two layers in the network explained yesterday are linear. These
ended up having a large amount of parameters. By now testing, the network
seems to be very sensitive to overfitting. At some point the training
loss improves, while the validation loss and my metric ends up increasing.
I have therefore played around with regularizing the network. In one case
it worked better than before, but not as good as the linear network.

# Day 45 [2019-03-04]
Continued looking at CNN architechtures for my problem, trying various
tweaks. It never performed as good as a model with only linear layer. In
the end I started looking into deep kernel learning for possible better
handling the final output, which should be a probability distribution.

# Day 46 [2019-03-05]
Watched week 2 videos of AI for all on coursera. Still quite basic. At times
some nice examples, tho.

# Day 47 [2019-03-06]
Worked through the first tutorial on regression with gpytorch and then one on
combining deep learning and Gaussian processes. The deep learning tutorial used
the UCI elevator dataset. Using a simple network with four linear layers, it 
tried to predict the dependent variable. Instead of just giving back an answer,
it gives a full probabilty distribution. In the figure below, the left panel
shows a scatter plot with the label and prediction when combining a deep learning
model and a Gaussian process. The prediction is simply the mean of the returned
distribution. Since the scatter looked relatively large, I also tried using a
standard neural networked trained using the mean squared error loss. This result
is shown in the right panel. The GP+deep learning loss look as good as the normal
deep learning result. Note this is not a completely rigorous comparison, since I
have not properly adjusted the number of epochs. It at least shows that the
large scatter is probably due to the problem or the network size, not the Gaussian
processes.

![gaussian process](https://github.com/marberi/100days/blob/master/gp_comparison.png)

# Day 48 [2019-03-07]
Looked into problems with a network we used for regression. Using a creative
loss function, I got some intitial interesting results last semester. Looking
at this again, the there is problems with the stability. At times the network
train and give a good results, while other times the metric is 10 times higher.
At least I managed to reproduce this issue, which a student found. It looks like
the network is overfitting. Not the question is how to fix this.

# Day 49 [2019-03-08]
Figured out a good solution. Adding an additional constraints on the returned
results made the network stable and giving quite good results. Answering how 
good will be the next step. I also looked more into the deep kernel learning.
The next step is construct my own example and apply a network to understand 
better how it works. So far I managed to generate some simple simulations.

# Day 50 [2019-03-09]
The first test today is constructing a network which can measure the parameters
of a parabola. This turned out to be extremely simple. Just creating a small
linear network, it trained very quickly to find the correct result. Below is
one example of the input and the resulting fit. The data had a SNR of 10 and
the loss fuction was just the MSE loss of the recovered parameters. This is a
good example of not needing to be too advanced to get results.

![parabola fit](https://github.com/marberi/100days/blob/master/parabola_fit.png)

# Day 51 [2019-03-10]
Finished the AI for everyone course. Quite some unexpected hours of watching
today. It is a recommended course, especially if starting out with AI. For a
technical person, it would not replace any of the more advanced courses where
you learn how to code. It puts AI in a larger perspective and give ideas on
how to use AI within a company and the benefit for society.

# Day 52 [2019-03-11]
One topic which I previously have been looking at is pretraining with
simulations. Since the training set is small, I plan to pretrain on simulations
before tweaking the network with real data. Quite some time ago this seemed
to work well. However, recently this was giving worse results than not
pretraining at all. These latest simulations was quite large, about one
million galaxies and based on galaxy evolution models. After not working, I
tried varying parameters like the distribution of galaxy ages, star formation
and metallicities. No success. What actually made a difference was adding some
noise to the galaxies. Pretraining with moderate amount of noise made sure
the solution was not overfitting. And now the result is better after pretraining
on simulations. It seems a bit sensitive to not pretraining for too long. Anyway,
at least one step forward.

# Day 53 [2019-03-12]
Thought the problem with flux estimation was already solved. Apparently there is
still a remainding bias. Tried rewriting the code in the process of looking into
this. No luck so far. At least I ended up with a rewritten and better code.

# Day 53 [2019-03-13]
Continued a bit looking at the flux estimation problem. From one constribution
to the loss, then I expected an unbiased estimate. However, it is not quite
clear what happens when combining multiple loss terms together. At night I
watched the following videos

![Geoffrey Hinton: The Foundations of Deep Learning](https://www.youtube.com/watch?v=zl99IZvW7rE) and 
![Yoshua Bengio: The Rise of Artificial Intelligence through Deep Learning](https://www.youtube.com/watch?v=uawLjkSI7Mo)

The first one was particulary nice. It included a discussion on the number of
training samples compared to the number of parameters. Also it mentioned how
a neural network can give better predictions than its labels.
