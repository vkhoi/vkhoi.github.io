---
layout: post
date: 2017-04-20
---
* #table of content
{:toc}

### 1. Introduction
**Discrete Cosine Transform (DCT)** is an algorithm that takes an input signal and represents it using an orthonormal basis. The DCT is similar to the **Discrete Fourier Transform (DFT)**. However, while the DFT's basis functions are in complex exponential form (\\(e^{ix}\\)), which can be decomposed into both cosine and sine functions, the DCT only depends on cosine functions to represent the signal. The DCT can also be thought as a **transform coding** operation that attempts to decorrelate the input signal into independent terms, thus reduces the redundancy in the data and provides compression capability.

Since image is a kind of signal in the spatial domain, the DCT can also be used to compress image. It is important to know that DCT is widely adopted in different coding standards, including the famous **JPEG** for image and **MPEG** for video.\\
\\(\\)
### 2. Why transform coding?
In an image, pixels in a small region tend to be correlated with each other. The same thing happens with video, where consecutive video frames exhibit correlation between neighboring pixels. Transform coding shall map the image in the spatial domain into uncorrelated coefficients in the frequency domain in order to reduce the redundancy. It should be noted that transform coding is a **lossless operation**, since it is possible to perfectly reconstruct the original image from the uncorrelated terms. Suppose the image contains \\(N\\) pixels, the DCT also uses exactly \\(N\\) coefficients to represent it, which shows that transform coding does not help in compressing at all, so why bother using it?

The ability to compress image of DCT stems from the fact that the human eye is unable to perceive some visual information that are subsumed in the DCT terms. Therefore, it becomes possible to remove some DCT terms without introducing any visual artifacts. This is also known as the **quantization** step.\\
\\(\\)

### 3. 1-dimensional Discrete Cosine Transform
Suppose we are given a real finite-length signal with \\(N\\) terms \\(x[n]\\) (or vector \\(\mathbf{x} \in \mathbb{R}^N\\)), we can use the DCT with \\(N\\) basis vectors \\(\psi_{k | k = 0, ..., N-1}\\) defined as follows

$$\psi_k[n] = \alpha[k] \cos{\Big(\frac{\pi(2n+1)k}{2N}\Big)} \text{ , for } n = 0, 1, ..., N-1 $$

where

$$\begin{align*}\alpha[k] &=
\begin{cases}
	\sqrt{\frac{1}{N}} & \quad \text{if } k = 0 \\
	\sqrt{\frac{2}{N}} & \quad \text{if } k = 1, ..., N-1
\end{cases} \\\\
\psi_k &= k \text{-th basis vector}
\end{align*}$$

Following is the illustration of some basis vectors.

![img1](/assets/dct/dct_example1.png){: .center-image}
**Figure 1.** Some 1-dimensional DCT basis vectors for \\(N = 64\\). We ignore the constant scaling factor \\(\alpha[k]\\). From right to left, from top to bottom are visualization of the \\(k\\)-th basis vector where \\(k = 0, 1, 2, ..., 6\\). Any finite-length signal with length \\(64\\) can be represented using this set of \\(64\\) cosine vectors.

One important property of the DCT is that its vectors produce an **orthonormal basis**.

We can then transform \\(x[n]\\) onto the above orthonormal basis. Since the basis vectors are orthonormal, the coefficient \\(y[k]\\) associated with each basis vector \\(\psi_k\\) can be calculated by simply using the inner-product of \\(x[n]\\) with the basis.

$$y[k] = \alpha_k \sum_{n=0}^{N-1}x[n]\cos{\frac{\pi (2n+1)k}{2N}}$$

The inverse transformation is similar

$$x[n] = \sum_{k=0}^{N-1}\alpha[k] y[k] \cos{\frac{\pi (2n+1)k}{2N}}$$

Both the transformation and its inverse version can be done using matrix multiplication. Define vector \\(\mathbf{x}\\) and \\(\mathbf{y}\\) as

$$\mathbf{x} =
\begin{bmatrix}
	x[0] \\
	x[1] \\
	\vdots \\
	x[N-1]
\end{bmatrix}\qquad
\mathbf{y} =
\begin{bmatrix}
	y[0] \\
	y[1] \\
	\vdots \\
	y[N-1]
\end{bmatrix}$$

And the DCT matrix \\(\mathbf{C}\\) as

$$\mathbf{C} =
\begin{bmatrix}
	c[0, 0]		& c[0, 1]	& \dots 	& c[0, N-1] \\
	c[1, 0] 	& c[1, 1] 	& \dots 	& c[1, N-1] \\
	\vdots 		& \vdots	& \ddots	& \vdots \\
	c[N-1, 0]	& c[N-1, 1]	& \dots 	& c[N-1, N-1]
\end{bmatrix}$$

where \\(c[k, n] = \alpha[k] \cos{\Big(\frac{\pi (2n+1)k}{2N}\Big)}\\).

Then \\(\mathbf{y} = \mathbf{Cx}\\), and \\(\mathbf{x} = \mathbf{C}^T\mathbf{y}\\) (since \\(\mathbf{C}\\) is an orthonormal matrix, so \\(\mathbf{C}^T\mathbf{C} = \mathbf{I}\\)).\\
\\(\\)
### 4. 2-dimensional Discrete Cosine Transform
As we are interested in using the DCT with image, we shall extend the aforementioned 1-dimensional DCT to 2-dimensional as follows

$$\psi_{k, l}[n, m] = \alpha[k] \alpha[l] \cos{\Big(\frac{\pi(2n+1)k}{2N}\Big)} \cos{\Big(\frac{\pi(2m+1)l}{2N}\Big)}$$

where \\(n, m = 0, 1, ..., N-1\\), and \\(\psi_{k, l} = \\) the \\([k, l]\\)-th basis vector. For the sake of simplicity, let's just assume the image's size is \\(N \times N\\).

Given an image \\(\mathbf{X}[n, m]\\), the coefficients \\(\mathbf{Y}[k, l]\\) associated with each basis vector is

$$\mathbf{Y}[k, l] = \alpha[k] \alpha[l] \sum_{n=0}^{N-1} \sum_{m=0}^{N-1} \mathbf{X}[n, m] \cos{\frac{\pi (2n+1)k}{2N}} \cos{\frac{\pi (2m+1)l}{2N}}$$

Or in vectorized form: \\(\mathbf{Y} = \mathbf{CXC}^T\\) and \\(\mathbf{X} =  \mathbf{C}^T\mathbf{YC}\\). It should be noted that this vectorized form can only be used when the image's height and width are equal.

Following is the illustration of the 2-dimensional DCT basis vectors. As they are in 2D, we shall convert their values in the frequency domain to the spatial domain, where neutral grey represents zero, black represents negative, and white represents positive amplitude.

![img2](/assets/dct/dct_example2.png){: .center-image}
**Figure 2.** Some 2-dimensional DCT basis vectors for \\(N = 64\\). We ignore the constant scaling factor \\(\alpha[k], \alpha[l]\\). From right to left, from top to bottom are visualization of the \\([k, l]\\)-th basis vector where \\(k, l = 0, 1, 2, ..., 7\\). Any image with size \\(64 \times 64\\) can be represented using this set of \\(64 \times 64\\) cosine vectors.

Now that we got a hold of the mechanism of the DCT. Next, we shall explore an interesting property that makes it possible to do quantization on the coefficients and compress an image.\\
\\(\\)
### 5. Energy Compaction of the DCT
The efficiency of a transformation coding is based on its ability to transform the input signal into as few coefficients as possible. Therefore, it becomes feasible to perform significant quantization on those with small amplitudes and achieve compression. Following are examples of the DCT coefficients computed on some images.

![img3](/assets/dct/dct_example3.png){: .center-image}
![img4](/assets/dct/dct_example4.png){: .center-image}
![img5](/assets/dct/dct_example5.png){: .center-image}
**Figure 3.** On each row: left - original image with size \\(250 \times 250\\); center - the first \\(50 \times 50\\) DCT coefficients (also the \\(50 \times 50\\) coefficients with lowest frequency); right - reconstruct the image using the first \\(100 \times 100\\) coefficients.

It can be seen from the above figure that the coefficients with high amplitude are not only compacted closely together but they are also distributed around the top-left region. Thus, it is intuitive to say that by using only the low frequency DCT basis vectors, we are still able to reproduce the image. It turns out that the coefficients with low frequency represent the general shape and pattern of the image, while the coefficients with high frequency represent texture, edge, and corner. So if the image we are working on is rich in texture, quantizing the high frequency terms may lead to ugly result.\\
\\(\\)
### 6. How DCT is used in JPEG
As mentioned above, quantizing high frequency terms will fail with image that has a lot of texture. In order to overcome this, JPEG divides the input image into non-overlapping square sub-blocks (\\(8 \times 8\\)) and perform DCT transformation on each individual sub-block. This idea comes from the fact that pixels in a local region are highly correlated, so these divided sub-blocks would not contain a whole bunch of different texture, the coefficients will be compacted in the low frequency region, and the DCT will work very well with each sub-block. Moreover, dividing the image into smaller blocks also help reduce the total time complexity. Following is the quantization table that JPEG uses for each sub-block.

![img8](/assets/dct/dct_example8.png){: .center-image}
**Figure 4.** The JPEG quantization table. The sub-block's DCT coefficients are divided by the corresponding number in the table (component-wise).

The quantization process reduces the quality of the image, and it explains why the image's quality becomes worse when you keep saving a JPEG image over and over. It can be seen from the above table that the high frequency region (the bottom-right conner) contains very large number, which makes the high frequency DCT coefficients likely become zero.

JPEG also applies zero run-length coding when storing the DCT coefficients, since the high frequency coefficients are likely zero. JPEG goes in the zig-zag order starting from the top-left of the sub-block, and stops when it reaches a point where all the cofficients after it are zero. After that, JPEG shall apply Huffman coding in order to compress the remaining DCT values.

![img6](/assets/dct/dct_example6.png){: .center-image}
**Figure 5.** The zig-zag order of JPEG. This is also the order from low to high frequency of the coefficients. 

\\(\\)
### 7. Code
The SciPy package contains the *fftpack* with the *dct* function that computes the DCT coefficients for a given signal. Followng is the code to compute the DCT coefficients, by using the SciPy package and by using matrix multiplication.

```python
def get_2d_dct(im, use_scipy=False):
    # Flag to indicate whether to use the SciPy package.
    if use_scipy:
        return fftpack.dct(fftpack.dct(im.T, norm='ortho').T, norm='ortho')
    
    # Get the signal's length.
    N = im.shape[0]
    
    # Compute the constant scaling factor alpha.
    a = [(1/N)**0.5 if k == 0 else (2/N)**0.5 for k in range(N)]
    
    # Create matrix C that holds the basis cosine vectors.
    C = np.zeros([N, N])
    for k in range(N):
        for n in range(N):
            C[k][n] = a[k]*math.cos(math.pi*(2*n+1)*k/2/N)
    
    # Transforms the input image onto this basis.
    res = np.dot(C, im)
    res = np.dot(res, C.T)
    
    return res
```

The inverse DCT transform is similar.

```python
def get_2d_idct(dct, use_scipy=False):
    # Flag to indicate whether to use the SciPy package.
    if use_scipy:
        return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

    # Get the signal's length.
    N = dct.shape[0]
    
    # Compute the constant scaling factor alpha.
    a = [(1/N)**0.5 if k == 0 else (2/N)**0.5 for k in range(N)]
    
    # Create matrix C that holds the basis cosine vectors.
    C = np.zeros([N, N])
    for k in range(N):
        for n in range(N):
            C[k][n] = a[k]*math.cos(math.pi*(2*n+1)*k/2/N)
    
    # Inverse transform.
    res = np.dot(C.T, dct)
    res = np.dot(res, C)
    
    return res
```

The aforementioned JPEG order to choose the lowest frequecy coefficients is implemented as follows.

```python
# Get next cell in the JPEG order.
def get_next_cell(cell, N):
    # r: row index; c: column index; d: direction we are going - 0 if going down, 1 if going up.
    r, c, d = cell
    if r == 0 and c % 2 == 0:
        c += 1
        d = 0
        if c >= N:
            r += 1
            c = N-1
    elif c == 0 and r % 2 == 1:
        r += 1
        d = 1
        if r >= N:
            c += 1
            r = N-1
    else:
        if d == 0:
            r += 1
            c -= 1
        else:
            r -= 1
            c += 1
    return (r, c, d)
```

The main processing code: we will perform the DCT transformation on an input image and reconstruct it using \\(20, 40, 60, ..., 2000\\) coefficients. 

```python
# Input image in grayscale.
IMAGE_NAME = 'data/messi.jpg'
im = cv2.imread(IMAGE_NAME, 0)

# Get image size. For simplicity, we only work with square image.
N = im.shape[0]

# Compute the DCT coefficients.
dct = get_2d_dct(im, use_scipy=True)

fig = plt.figure(figsize=(14, 14))

# Starting from 0 coefficient, at each step, we choose 20 more coefficients.
step = 20

# The first cell starts from (0, 0).
cell = (0, 0, 0)

# Quantized DCT coefficients.
# We choose to keep the lowest coefficients which is similar to that of JPEG. 
quantized_dct = np.zeros([N, N])

# Index of the image for plotting.
idx = 0

# 100 steps.
for i in range(100):
    idx += 1
    
    # Choose additionally more $step$ coefficients.
    for j in range(step):
        cell = get_next_cell(cell, N)
        quantized_dct[cell[0]][cell[1]] = dct[cell[0]][cell[1]]
    
    # Reconstruct.
    reconstructed = get_2d_idct(quantized_dct, use_scipy=True)
    
    # Plot.
    ax = fig.add_subplot(10, 10, idx)
    ax.axis('off')
    ax.imshow(reconstructed, cmap=plt.cm.gray)
```

Here is the plot of the result.

![img7](/assets/dct/dct_example7.png){: .center-image}
**Figure 6.** The reconstruction process of an image. From top to bottom, left to right: choose \\(20, 40, 60, ..., 2000\\) lowest frequency coefficients to reconstruct.