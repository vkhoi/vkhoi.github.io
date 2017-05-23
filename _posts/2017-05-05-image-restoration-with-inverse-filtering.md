---
layout: post
title: "Image Restoration with Inverse Filtering"
date: 2017-05-05
---
* #table of content
{:toc}

### 1. Introduction
Image restoration are techniques that are used to fix distortion in images. In this post, I shall present a simple technique called **inverse filtering** to remove well-known distortion in image such as out-of-focus blur and blur due to motion when capturing. It should be noted that inverse filtering is not a correct way to use for image restoration, but it is a basic knowledge in the field and it is good to know about it.

### 2. What is Image Restoration?
Captured images are usually not perfect but **distorted** due to numerous reasons: out-of-focus, shaky hand, sensor noise, environmental conditions (reflected light). Image restoration refers to techniques that aim to reduce these distortion (or even "undo" them) and improve the image's quality. Following are examples of two well-known problems when taking picture: out-of-focus and shaky hand. For illustration purpose, these 2 images are produced by using simple filter.

![img1](/assets/inverse-filtering/text-blur.png){: .center-image}
**Figure 1.** left: original image; right: blurred image using a \\(11 \times 11\\) filter with all 1's.

![img2](/assets/inverse-filtering/camera-man-motion-blur.png){: .center-image}
**Figure 2.** left: original image; right: motion-blurred image micmicking the horizontal movement of the sensor when capturing (produced using a \\(31 \times 31\\) filter with one row equals to 1).\\
\\(\\)

The process during which the image is distorted is called the **image degradation** process. This process is followed by the **image restoration** step, where we will perform techniques to reverse the distortion that has been applied. However, things are not going to be easy, as the degradation step also introduces additional **noise** into the image. Thus, it becomes impossible to perfectly reconstruct the original image, and the best we can do is to produce an approximated version of it. Following is the diagram for a common model of an image restoration system.

![img3](/assets/inverse-filtering/diagram.png){: .center-image}
**Figure 3.** Model of an image restoration system. Notice that there are noisy values introduced between the degradation and restoration process.

In this blog, we are interested in degradation function that is **linear and space-invariant**, which is analogous to the term **linear time-invariant** system in signal processing. Based on this assumption, the degradation step can be modeled as a **convolution operator** between the original image and a filter. We will now explore using the **inverse filter** of the degradation function to reproduce the original image.\\
\\(\\)

### 3. Inverse Filtering
The image degradation process can be modeled using the following formula

$$g(x, y) = f(x, y) * h(x, y) + \eta(x, y)$$

where \\(g(x, y)\\) is the distorted image due to the degradation function \\(h\\) plus some noise \\(\eta\\), and \\(*\\) is the convolution operator. To introduce the inverse filtering, let's assume for now that there is no noise occured in this process, that is 

$$g(x, y) = f(x, y) * h(x, y)$$

By taking the **Fourier transform** on both side of the equation, we are able to move from the spatial domain to the frequency domain. One interesting property that we are going to exploit is that in the frequency domain, the **convolution operator becomes multiplication**. Thus,

$$G(x, y) = F(x, y) \cdot H(x, y)$$

where \\(G(x, y), F(x, y), H(x, y)\\) are respectively the Fourier transform of the three original images and filters. We can then simply compute the Fourier transform \\(F\\) and **inverse** it to get the original image.

$$F(x, y) = \frac{G(x, y)}{H(x, y)}$$

This is what we call **inverse filtering**. If we manage to estimate what kind of filter has been applied and its parameters, we can easily compute its inverse \\(\frac{1}{H(x, y)}\\). However, this formula involving division may lead to **computational problem** because there is no guarantee that \\(H(x, y) \neq 0\\) for all \\(x\\) and \\(y\\). For example, the simple blur filter is a low-pass filter and its Fourier transform contains a lot of \\(0\\) in the high frequency terms. Therefore, we introduce a **pseudo-inverse** filter as follows:

$$H_{inv}(x, y) =
\begin{cases}
	1 / H(x, y) & \quad \text{if } |H(x, y)| \geq \text{threshold} \\
	0 & \quad \text{otherwise}
\end{cases}$$

and

$$\hat{F}(x, y) = G(x, y) \cdot H_{inv}(x, y)$$

With this pseudo-inverse filter, we can try different values of threshold until we produce a satisfactory result. So even though it is assumed that there is no noise occurred in the degradation, we are still unable to reconstruct the original image due to computational problem.

Now, suppose that there really is noise introduced in the degradation process. Then,

$$\begin{align*}
g_{noise}(x, y) &= g(x, y) + \eta(x, y) \\
G_{noise}(x, y) &= G(x, y) + N(x, y)
\end{align*}$$

and

$$\begin{align*}
\hat{F}(x, y) &= \frac{G_{noise}(x, y)}{H(x, y)} \\
&= G(x, y) \cdot H_{inv}(x, y) + N(x, y) \cdot H_{inv}(x, y)
\end{align*}$$

Notice that in the last equation, the noise values are greatly amplified by \\(\frac{1}{H(x, y)}\\) when \\(H(x, y)\\) is small. Therefore, if there is noise in the degradation process, the noise terms will be **greatly increased** by the inverse filter and it will intensively distort the image. This is the reason that inverse filtering is **NOT** a good technique for image restoration. There is one other technique called **Wiener Filtering** that uses the same idea with inverse filtering but is able to overcome the noise problem. This technique will be presented in another post.

In summary, inverse filtering technique can be used when the following 2 ideal conditions are met:
* The filter and its parameter can be estimated. One way to estimate the filter is to forward the **delta** signal \\(\delta\\) through the degrading model. Thanks to the convolution property that \\(\delta * H = H\\), we can figure out the filter \\(H\\).
* There is no noise introduced in the degrading process.

![img4](/assets/inverse-filtering/result1.png){: .center-image}
**Figure 4.** left: original image; center: distorted image; right: restored image using inverse filtering.

![img5](/assets/inverse-filtering/result2.png){: .center-image}
**Figure 5.** left: original image; center: distorted image; right: restored image using inverse filtering.
\\
\\(\\)
### 4. Code
Import the necessary modules.

```python
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import fftpack
from scipy import signal
from scipy.ndimage.filters import convolve

%matplotlib inline
```

Auxiliary method to display grayscale image.

```python
# Display grayscale image.
def displayImage(img):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap='gray')
```

First, we will input an image and display it.

```python
# Input image.
IMAGE_NAME = 'data/camera-man.jpg'
img = cv2.imread(IMAGE_NAME, 0)

# Display image.
displayImage(img)
```
![img6](/assets/inverse-filtering/code-camera-man.png){: .no-mg}

We create a motion blur filter and apply it onto the image.
```python
# Create motion blur filter with size K.
K = 31
h = np.zeros((K, K))
h[K//2,:] = np.ones(K) / K

# Apply onto the image.
g = convolve(img, h, mode='wrap')
displayImage(g)
```
![img7](/assets/inverse-filtering/code-camera-man-blur.png){: .no-mg}

Next, we pad the filter with \\(0\\) so that it has the same size as the input image. Because the convolution operator in image processing and the **convolve** function that we used above uses the middle element of the filter as the center, while in signal processing the convolution operator centers at the first element of the filter, we need to shift the filter \\(H\\) before performing Fourier transform so that the filter's center lies at the first element.

```python
M = g.shape[0]
N = g.shape[1]

# Pad the filter with 0.
h_pad = np.zeros((M, N))
h_pad[(M-h.shape[0])//2:(M-h.shape[0])//2+h.shape[0],(N-h.shape[1])//2:(N-h.shape[1])//2+h.shape[1]] = h

# Shift the filter so that its center lies at the first element, aka H[0,0].
h_pad = scipy.fftpack.ifftshift(h_pad)
```

Next, we compute the Fourier transform.

```python
G = scipy.fftpack.fft2(g)
H = scipy.fftpack.fft2(h_pad)
```

And perform inverse filtering.

```python
# Set threshold to 0.005.
threshold = 0.005

# Create array to store the pseudo-inverse filter.
H_inv = np.zeros(G.shape, dtype=np.complex)

for r in range(H.shape[0]):
    for c in range(H.shape[1]):
    	# Compute the magnitude and compare it with threshold.
        mag = np.abs(H[r,c])
        if mag <= threshold:
            H_inv[r,c] = 0
        else:
            H_inv[r,c] = 1.0 / H[r,c]

# Compute the approximated Fourier transform of the original image.
F = G * H_inv

# Inverse the Fourier transform to get the original image.
f = scipy.fftpack.ifft2(F)
f = np.abs(f)

# Some values are larger than 255, so we clamp it to 255.
for r in range(f.shape[0]):
    for c in range(f.shape[1]):
        if f[r,c] > 255:
            f[r,c] = 255
displayImage(f)
```
![img7](/assets/inverse-filtering/code-camera-man-result.png){: .no-mg}

Following are the results with different threshold values.

![img8](/assets/inverse-filtering/thresholds.jpg){: .center-image}
**Figure 6.** From left to right are the results with the threshold value set at \\(0.001, 0.01, 0.1\\) respectively.