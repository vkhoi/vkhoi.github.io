---
layout: post
date: 2017-05-21
---
* #table of content
{:toc}

### 1. Introduction
Image segmentation is a fundamental problem in image processing. The main goal of image segmentation is to isolate out the objects of interest in the input image, then use them for other  purposes such as object recognition, object tracking, or compositing them into other backgrounds.

Image segmentation can be **done automatically** on simple images, where the interested object is clearly visible, center aligned, or distinctly recognizable from the background. However, for complicated images, such as ones with cluttered scenes, multiple potentially interested objects, image segmentation is difficult to automatically implemented. Therefore, it becomes essential to integrate **user interaction** into the process in order to differentiate between the object of interest (foreground) and the background. One possible user interaction is **drawing scribbles** onto the image, with red and green color to signify the foreground and background respectively. This note shall discuss an image segmentation technique with the help of user interaction. This technique is presented in the paper **[here](http://www.tc.umn.edu/~baixx015/Xue_IJCV.pdf)** of **Xue Bai** and **Guillermo Sapiro**.

![img1](/assets/image-segmentation-user-interaction/james-result.png){: .center-image}
**Figure 1.** Image segmentation result on an example image.

\\(\\)
### 2. General Framework
There are 3 main steps in the framework. The first step is to perform **probability density estimation** of the color space of the image. In other words, we estimate the probability distribution function of the color of every pixel in the image. The second step is to calculate the **geodesic distance** between every pixel in the image and their nearest scribble. Using this distance, we can deduce whether a pixel belongs to the foreground or the background. The third step is an **enhancing** step, in which we automatically add new scribbles at places that we know for sure they are foreground or background, then perform step 1 and step 2 again. Details of the steps are presented in the next sections.

![img2](/assets/image-segmentation-user-interaction/framework.jpg){: .center-image}
**Figure 2.** The general framework.

\\(\\)
#### 2.1 Probability Density Estimation of the Color Space
#### 2.1.1 The Histogram
The original problem is as follows: given a dataset with \\(N\\) data points, we need to give an estimation of the **probability density function (PDF)** of the dataset **directly** from these points, without using any **prior information on the underlying distribution**. This is also called the non-parametric approach to the density estimation problem.

One simple solution is to use the **histogram**. By counting the frequency of data points occurring in the dataset, we can calculate the probability of any arbitrary value. However, the PDF estimated from the histogram is a discrete function with discontinuties. Any values that do not occur in the dataset shall be estimated with probability \\(0\\). The histogram also relies on our choices of the bins' width and the starting location of the first bin. Using any other values of these two parameters will result in another shape of the histogram, which leads to difficulties in interpreting the dataset (Figure 2). Moreover, histogram severely suffers from the **curse of dimensionality**. When the dataset is multidimensional, it is difficult to use the histogram method because it produces a large number of bins. Therefore, the histogram is inappropriate to be applied in this problem. We move on to another one, which is also a non-parametric method similar to the histogram, the **kernel density estimation**.

![img3](/assets/image-segmentation-user-interaction/histogram-shifted.png){: .center-image}
**Figure 3.** Effect on the visualization of the histogram when we change the start location of the first bin (image taken from [here](http://scikit-learn.org/stable/modules/density.html)).

![img4](/assets/image-segmentation-user-interaction/histogram.png){: .center-image}
**Figure 4.** An example of using the histogram approach to estimate the PDF of a dataset (image taken from [here](http://research.cs.tamu.edu/prism/lectures/pr/pr_l7.pdf)).

\\(\\)
#### 2.1.2 Kernel Density Estimation
The idea of the kernel density estimation method is that each example in the dataset will generate a distribution around it. The sum of these distributions are then aggregated to produce the overall distribution of the whole dataset. Suppose our dataset contains \\(N\\) data points in the \\(D\\)-dimensional space, the probability of occurring of an arbitrary input vector \\(\mathbf{x}\\) is calculated using the following formula

$$\begin{equation}
p_{kde}(\mathbf{x}) = \frac{1}{Nh^D}\sum_{n=1}^{N}K(\frac{\mathbf{x}-\mathbf{x}^{(n)}}{h})\end{equation}$$

where \\(h\\) is a value that represents the width (or spanning) of the distribution of each data point, which is also called the **bandwith**, and \\(\mathrm{K}\\) is the kernel function. The kernel function shows how all data points are affecting the probability of the input value.

Imagine we are in the \\(3\\)-dimensional space, and around each data point \\(\mathbf{x}\\) is a cube with side length \\(h\\) centered at \\(\mathbf{x}\\). The kernel function \\(\mathrm{K}\\) is defined as follows

$$ K(u) =
\begin{cases}
    1 & \quad \text{if } |u_j| \leq 1/2 \text{ for } j = 1, 2, 3 \\
    0 & \quad \text{otherwise }
\end{cases}$$

Using this kernel function with the formula for \\(\mathrm{p_{kde}}(\mathbf{x})\\), we are actually counting the number of cubes that the input data point \\(\mathbf{x}\\) lies inside (Figure 4). Hence the more number of cubes, the larger the probability.

![img5](/assets/image-segmentation-user-interaction/cube.png){: .center-image}
**Figure 5.** Cube centered at \\(\mathbf{x}\\) with side length \\(h\\).

The above kernel function is called the **Parzen window**, and it can be scaled upto higher dimensions. The Parzen window also resembles the histogram, but with its bins located at each data point instead of equally-spaced bins. However, the Parzen window suffers from discontinuties, and all data points \\(\\mathbf{x}^{(n)}\\) contributes equal weights to the result probability regardless of their distance towards the center. Thus, we will apply the **Gaussian kernel** instead of the Parzen window in our problem. The Gaussian kernel provides smoothing property and can alleviate the above problem of the Parzen window. An example of using the Gaussian kernel to estimate the PDF of a dataset is provided in figure 5.

$$ K_{Gaussian}(u) = \frac{1}{\sqrt{2\pi}^D}e^{-\frac{u^2}{2}}$$

![img6](/assets/image-segmentation-user-interaction/gaussian-kernel.png){: .center-image}
**Figure 6.** The red curve represents the Gaussian distribution centering at each data point. The blue curve is the aggregation of all the individual curves, produces the estimated PDF (image taken from [Wikipedia](http://research.cs.tamu.edu/prism/lectures/pr/pr_l7.pdf)).

\\(\\)
#### 2.1.3 Gaussian Kernel Density Estimation of the Color Space

Let \\(\Omega_F\\) and \\(\Omega_B\\) be the set of pixels that belong to the foreground and background respectively, according to the scribbles drawn on the image. On each set, we run the kernel density estimation on the color space of its pixels and produce its PDF. From the PDF, the probability of a pixel with color \\(\mathbf{c}\\) to belong to the foreground is

$$P_F(\mathbf{c}) = \frac{PDF_F(\mathbf{c})}{PDF_F(\mathbf{c}) + PDF_B(\mathbf{c})}$$

where \\(PDF_F(\mathbf{c})\\) is the PDF of color vector \\(\mathbf{c}\\) that we estimated. This probability value will be used in the next step to calculate the geodesic distance between each pixel and the scribbles. Following is an example result of \\(P_F\\), illustrated using a heat map. The redder the color of a pixel, the more likely that pixel belongs to the foreground based on the PDF of the color space.

![img7](/assets/image-segmentation-user-interaction/james-james.png){: .center-image}
**Figure 7.** Example result of \\(P_F\\), illustrated using a heat map. The redder the color of a pixel, the more likely that pixel belongs to the foreground based on the PDF of the color space.

\\(\\)
#### 2.2 Geodesic Distance Calculation

An image can be modeled as a graph, in which each pixel is a vertex and the edges connect adjacent pixels together. Two pixels are considered adjacent if the Manhattan distance between them is less than or equal to \\(1\\). Using the probability value \\(P_F(\mathbf{c})\\) that we calculated from the previous step, we define the cost of each edge connecting pixel \\(x\\) and \\(y\\) as \\(W(x, y) = \| P_F(\mathbf{c}_x) - P_F(\mathbf{c}_y) \|\\), where \\(\mathbf{c}_x\\) is the color vector of pixel \\(x\\). Using **Dijkstra algorithm** on this graph, we then can calculate the distance of the shortest path between every pixel and its nearest foreground scribble. Following is a heat map to represent the cost of the shortest path from all pixels to their nearest foreground scribble.

![img8](/assets/image-segmentation-user-interaction/james-foreground-cost.png){: .center-image}
**Figure 8.** Cost of shortest path from all pixels to their nearest foreground scribble. Notice that pixels at the foreground scirbble's location have darkest blue, since they themselves are the foreground scribble.

Similarly, using the probability value \\(P_B(\mathbf{c})\\) from the previous step, we can use Dijkstra to compute the length of the shortest path from all pixels to their nearest background scribble. Finally, for each pixel, we compare between the length of the shortest path from it to its nearest foreground and background scribble to find out which region it does belong. Following is the result of this step. Notice there are artifacts around the border. We can enhance the result using the approach in the next section.

![img9](/assets/image-segmentation-user-interaction/james-mid-result.png){: .center-image}
**Figure 9.** Result after applying the geodesic distance calculation to determine every pixel's region.

\\(\\)
#### 2.3 Enhancing

The idea of Xue Bai and Guillermo Sapiro is as follows. From the current connected components of the foreground, we can artificially insert a foreground scribble that wraps around the object of interest. Moreover, we can even insert new background scribble to wrap outside the object of interest with a wider margin. This idea is illustrated in the next figure.

![img10](/assets/image-segmentation-user-interaction/james-artificial-scribble.png){: .center-image}
**Figure 10.** Add artificial scribbles around the object of interest.

Finally, we re-run step 1 and step 2 again to get a better result without artifacts near the border.

![img10](/assets/image-segmentation-user-interaction/james-final.png){: .center-image}
**Figure 11.** Result after enhancing.
