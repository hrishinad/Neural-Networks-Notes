# 8. Batch Normalization and Autoencoders: Stability and Compression

> [!quote] "The goal is to turn data into information, and information into insight." — Carly Fiorina

In this chapter, we move beyond just tuning hyperparameters to architectural techniques that stabilize training and allow the network to learn efficient representations of data. We'll explore how to keep the "moving goalposts" of deep networks still with **Batch Normalization**, how to optimize learning with **Mini-batch Gradient Descent**, and how to compress data using the elegant architecture of **Autoencoders**.

---

## 8.1 Mini-batch Gradient Descent: The "Weekly Grocery" Strategy

Before we dive into normalization, we must understand the "cadence" of learning. In previous chapters, we often talked about "Gradient Descent" as a single concept, but in practice, it comes in three flavors.

### 8.1.1 The Analogy
Imagine you need to buy groceries for the month.

> [!abstract]
> - **Batch Gradient Descent (The Monthly Trip)**: You go once, buy everything in one massive haul. It's efficient (one trip), but your car is heavy, and you might get stuck in traffic for hours (Memory intensive, very slow updates).
> - **Stochastic Gradient Descent (The Single Item Trip)**: Every time you realize you need one thing (e.g., a lemon), you drive to the store. You make 100 trips. You get updates fast, but you're constantly driving back and forth, and the path is very erratic (Fast updates, but extremely noisy/noisy convergence).
> - **Mini-batch Gradient Descent (The Weekly Trip)**: You go once a week. You buy a reasonable amount. It's the "Goldilocks" zone—stable enough to be efficient, but frequent enough to make progress quickly.

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1400%2F1*bKSddSmLDaYszWllvQ3Z6A.png&f=1&nofb=1&ipt=fcf59055f39cd002655fa8fd079edc289819cb99dc912fb990e57b38dc9e4504)
### 8.1.2 The Process
Instead of calculating the gradient for the entire dataset of size $m$, we split the data into small batches of size $B$ (typically $32, 64, 128, \dots, 1024$).

1. Pick a mini-batch $(X^{\{t\}}, Y^{\{t\}})$.
2. Perform Forward Propagation on the mini-batch.
3. Compute the Loss $J^{\{t\}}$.
4. Perform Backward Propagation to get gradients.
5. Update Weights: $W = W - \alpha \cdot dW$.

### 8.1.3 Benefits
- [p] **Speed**: Vectors fit into GPU memory better.
- [p] **Convergence**: The "noise" in mini-batches can actually help the model jump out of local minima.
- [p] **Memory**: It allows training on datasets much larger than the available RAM.

---

## 8.2 Batch Normalization: The "Moving Goalposts" Problem

As we go deeper into a network, the distribution of activations in later layers changes constantly because the weights of previous layers are updating. This is called **Internal Covariate Shift**.

> [!abstract] The "Cooking Class" Analogy
> Imagine you are a student learning to make a complex sauce. Your teacher (Layer 1) prepares the base, and you (Layer 2) add the final spices. 
> If the teacher keeps changing the saltiness of the base every day, you can never learn exactly how much spice to add. You are chasing "moving goalposts."
> **Batch Normalization (BN)** ensures the teacher always delivers a base with a consistent "saltiness" (mean and variance).

### 8.2.1 The Math of Batch Norm
For a given mini-batch, BN performs the following operations on the activations $Z$ of a layer:

1. **Calculate Mean**: $\mu_B = \frac{1}{m} \sum_{i=1}^m z^{(i)}$
2. **Calculate Variance**: $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (z^{(i)} - \mu_B)^2$
3. **Normalize**: $\hat{z}^{(i)} = \frac{z^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$ (where $\epsilon$ is a tiny constant for numerical stability).
4. **Scale and Shift**: $\tilde{z}^{(i)} = \gamma \hat{z}^{(i)} + \beta$

> [!warning] Why Gamma ($\gamma$) and Beta ($\beta$)?
> If we only normalized to mean $0$ and variance $1$, we might lose the expressive power of the layer (e.g., we might force all values into the linear region of a Sigmoid). $\gamma$ and $\beta$ are **learnable parameters** that allow the network to "undo" the normalization if that's what's best for learning.

![475](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fsubstackcdn.com%2Fimage%2Ffetch%2Ff_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep%2Fhttps%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5ed1a42c-83c6-4277-8eb7-f1330cf01785_810x1080.gif&f=1&nofb=1&ipt=ffa5cdde2aa0be61babd9d5bdaa2d5ae1c5dc35c6a6d9237b477c3b30b9e6213)

### 8.2.2 Batch Norm at Inference (Testing)
During testing, we don't have a "batch" to calculate $\mu$ and $\sigma$. 
- [u] **Solution**: We use a **Running Average** of the means and variances calculated during the training phase.

### 8.2.3 Advantages
- [p] **Faster Convergence**: You can use much higher learning rates.
- [p] **Reduces Sensitivity**: The model is less dependent on weight initialization.
- [p] **Slight Regularization**: The noise from the batch statistics adds a small "jitter" that helps prevent overfitting.

---

## 8.3 Autoencoders: The Art of Compression

An **Autoencoder (AE)** is a neural network designed to learn a compressed, distributed representation of its input, typically for the purpose of dimensionality reduction or feature learning.

> [!abstract] The "Executive Summary" Analogy
> Imagine you have a 1,000-page legal document (Input). You ask a junior lawyer (**Encoder**) to summarize the core arguments onto a single index card (**Bottleneck**). Then, you give that index card to a senior partner (**Decoder**) and ask them to reconstruct the entire 1,000-page document. 
> 
> To succeed, the junior lawyer cannot just pick random sentences; they must identify the "essence" of the case. If the senior partner can reconstruct a document that is 99% similar to the original, it means the index card captured the most vital features of the data.

### 8.3.1 Structure: The Hourglass
The AE consists of two main parts connected by a "bottleneck":

1. **The Encoder**: Compresses the input $X$ into a low-dimensional latent space $h$ (the bottleneck).
2. **The Bottleneck**: The compressed "summary" of the data. 
3. **The Decoder**: Reconstructs the input from the latent space to produce $\hat{X}$.

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%2Fid%2FOIP.k-kGR2Iw-0jO6vPmmt_UAAHaFt%3Fpid%3DApi&f=1&ipt=971a5a5f7a4befbdeb28018948a13ae9d250c92d07ac58cf7a8944f8ec3d87c5)
### 8.3.2 The Loss Function
Since the goal is to make the output $\hat{X}$ as close to the input $X$ as possible, we use **Reconstruction Loss**:
$$L(X, \hat{X}) = \|X - \hat{X}\|^2$$ (Mean Squared Error)

### 8.3.3 Implementation Notes & Pitfalls
- [!] **Over-complete Hidden Layers**: If the bottleneck is larger than the input, the network might just "copy-paste" the input without learning anything useful.
- [*] **Dimensionality Reduction**: AE is like a non-linear version of PCA (Principal Component Analysis).

```python title:"Simple Autoencoder in Keras"
from tensorflow.keras import layers, models

# Input size: 784 (28x28 images)
input_img = layers.Input(shape=(784,))

# Encoder: Compress to 32 dimensions
encoded = layers.Dense(32, activation='relu')(input_img)

# Decoder: Reconstruct back to 784
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# The Autoencoder Model
autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## 8.4 Practical Examples: Denoising and Compression

### 8.4.1 Image Compression
By forcing the network through a tiny bottleneck (e.g., compressing a 784-pixel image into 32 numbers), we force the model to learn the most important features (edges, curves, shapes) rather than individual pixels.

### 8.4.2 Case Study: MNIST Image Denoising
In a **Denoising Autoencoder (DAE)**, we intentionally corrupt the input but keep the original, clean image as the target. This forces the network to learn the "essence" of the data to reconstruct the signal from the noise.

#### Step 1: Data Preparation
First, we load the MNIST dataset and normalize the pixel values to the $[0, 1]$ range. Since we are using an **Artificial Neural Network (ANN)**, we flatten the $28 \times 28$ images into a $784$-dimensional vector.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets

# Load MNIST
(x_train, _), (x_test, _) = datasets.mnist.load_data()

# Normalize and Flatten (28*28 = 784)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))
```

#### Step 2: Injecting Synthetic Noise
We add Gaussian noise to our clean images. This creates the "corrupted" version of the data. To keep the images valid, we clip the pixel values back to the $[0, 1]$ range.

```python
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

# Ensure pixels stay in valid range
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```

#### Step 3: Building the Dense Autoencoder
This architecture uses simple `Dense` layers. The **Encoder** reduces the $784$ input features down to a "bottleneck" of $32$, forcing the network to learn only the most important features. The **Decoder** then tries to expand this back to $784$.

```python
# Bottleneck size
encoding_dim = 32 

input_img = layers.Input(shape=(784,))

# Encoder: Contracting to 32 neurons
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# Decoder: Reconstructing back to 784
decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

# Compile the model
autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

#### Step 4: Training for Reconstruction
We train the model to map the **noisy input** back to the **original, clean target**. This teaches the network to ignore the "jitter" and reconstruct the underlying digit.

```python
# Map Noisy Input -> Clean Target
autoencoder.fit(x_train_noisy, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
```

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fstatic.packt-cdn.com%2Fproducts%2F9781788629416%2Fgraphics%2FB08956_03_09.jpg&f=1&nofb=1&ipt=bf5fc7605621b7058f5160d26900ace1995771f605820f3bfacce4d4108dbb24)

---

## Summary
In this chapter, we tackled the stability of deep networks using **Batch Normalization** and the efficiency of learning with **Mini-batch Gradient Descent**. We also explored **Autoencoders**, which serve as the foundation for generative modeling and self-supervised learning by mastering the art of compression and reconstruction. 
 