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

![SVD separation] (people_walking.png)
