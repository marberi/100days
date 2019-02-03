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
