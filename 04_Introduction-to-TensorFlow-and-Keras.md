# 4. Introduction to TensorFlow and Keras

Up until now, we have built neural networks entirely from scratch using raw Python and NumPy. We hand-calculated the chain rule, explicitly managed weight matrices, and wrote our own training loops. 

While this mathematically rigorous approach is the *only* way to truly understand what a neural network is doing under the hood, nobody does this in the real world. Why? Because it's slow, prone to bugs, and extremely difficult to scale across GPUs.

Welcome to the world of Modern Deep Learning Frameworks.

---

## 4.1 TensorFlow vs. Keras

You will hear these two terms constantly interchanged in the industry. Let's clarify the difference.

### TensorFlow
TensorFlow is a massive, comprehensive open-source platform for machine learning developed by Google. At its core, it is an incredibly powerful engine designed to execute complex tensor mathematics across thousands of CPUs and GPUs seamlessly. 

However, raw TensorFlow 1.x was notoriously difficult to use. Writing a simple model required a lot of boilerplate code and steep learning curves.

### Keras
Keras is a high-level neural networks API written in Python. It was originally designed to run *on top* of various backend engines (like TensorFlow, CNTK, or Theano) to make building neural networks fast, user-friendly, and accessible.

> [!important] The Merger (TensorFlow 2.x)
> Because Keras was so universally loved by developers for its simple API, Google officially adopted it. With the release of TensorFlow 2.0, **Keras became the default, built-in high-level API for TensorFlow.** You no longer need to install them separately. You access Keras *through* TensorFlow.

> [!abstract] **Teacher's Analogy:** 
> Imagine **TensorFlow** as a massive, industrial V8 engine capable of rocketing a car to 300mph. It is raw, complicated, and dangerous to touch directly.
> Imagine **Keras** as the beautiful leather steering wheel, pedals, and dashboard. You don't need to rebuild the V8 engine to drive to the store; you just use the steering wheel! Keras allows us to harness the terrifying power of TensorFlow using simple, readable Python.

**Key Features of Keras:**
- **Ease of Use:** It drastically reduces the cognitive load and lines of code needed.
- **Modularity:** Neural networks are built by simply snapping together lego blocks (layers, optimizers, loss functions).
- **Extensibility:** You can easily write custom loss functions or layers if the built-in ones don't meet your needs.

---

## 4.2 The Fundamentals: Models and Layers

Before we write code, we must understand the two core concepts that the entire Keras framework is built around: Layers and Models.

### 1. What is a "Layer"?
In Keras, a **Layer** is the fundamental building block. It is a data-processing module that takes one or more tensors as input and outputs one or more tensors. Crucially, layers contain the network's **state**—the interconnected weights and biases (which Keras handles for you invisibly) that are updated during training. 

By stacking different types of layers, we can build architectures to process anything from simple spreadsheets to complex video feeds.

### 2. What is a "Model"?
A **Model** is simply a container or a directed acyclic graph that holds your Layers together. It defines the flow of data from the input layer(s) through the hidden layers to the output layer(s). The model is the object that you compile (give an optimizer/loss) and fit (train) on your data.

---

## 4.3 The Keras Sequential API

The most straightforward way to build a model in Keras is using the **Sequential** model. As the name implies, it allows you to build a network layer-by-layer in a simple, linear stack, where each layer has exactly one input tensor and one output tensor.

### The Lifecycle Flowchart

Let's visualize the exact 5-step lifecycle of creating and training a model using the Keras Sequential API.

![](keras-sequential-api.svg)

---

## 4.4 Building a Neural Network

Let's translate that flowchart into actual code. We will build a network similar to our Chapter 3 example.

### Step 1: Importing Libraries
Since Keras is integrated into TensorFlow, we import what we need directly from the `tensorflow` package.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

### Step 2 & 3: Initializing and Adding Layers
We instantiate the `Sequential` object, and then use the `.add()` method to stack our layers.

The most common layer we use for standard multi-layer perceptrons is the **Dense Layer**. 

### Understanding the Dense Layer
A Dense layer is "fully connected." This means exactly what we've been practicing in numpy: every single neuron in the current layer is connected via a weight to **every single neuron** in the previous layer.

![](dense-layer.svg)


```python
model = Sequential()

# Hidden Layer 1
model.add(Dense(units=64, activation='relu', input_shape=(11,)))

# Hidden Layer 2
model.add(Dense(units=32, activation='relu'))

# Output Layer
model.add(Dense(units=1, activation='sigmoid'))
```

> [!warning] The `input_shape` Parameter
> Notice that the *very first* layer requires an `input_shape` argument. Keras needs to know the shape of your incoming data (e.g., 11 features) to properly instantiate that initial weight matrix $W^{[1]}$. Subsequent layers figure out their shapes automatically!

---

## 4.5 The Functional API & The `Input` Class

The `Sequential` API is fantastic for beginners because it's simple: a straight line from Input A to Output B. However, the real world is rarely a straight line.

What if your model takes **two inputs** simultaneously (e.g., an image of a house *and* text describing the neighborhood) to predict the price? What if you want layers to branch out, or skip over each other (like in ResNet)? The `Sequential` model simply cannot do this.

This is where Keras' most powerful feature comes in: the **Functional API**.

### The `Input` Object
Instead of letting a wrapper manage the flow, the Functional API forces you to manually wire the outputs of one layer into the inputs of the next layer. 

To start this chain, you *must* explicitly define an `Input` object. This acts as an empty placeholder tensor that tells Keras exactly what shape of data is coming.

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 1. Provide an explicit Input tensor
inputs = Input(shape=(11,))

# 2. Wire the layers together sequentially like a chain
x = Dense(64, activation='relu')(inputs)  # Pass 'inputs' into this Dense layer
x = Dense(32, activation='relu')(x)       # Pass 'x' into this next Dense layer
outputs = Dense(1, activation='sigmoid')(x)

# 3. Define the start and end of the Model
model = Model(inputs=inputs, outputs=outputs)
```

Look closely at the syntax: `Dense(...)(previous_layer)`. We are treating the `Dense` class like a *callable function*, which is why this is called the Functional API!

> [!abstract] **Teacher's Analogy:**
> Think of `Sequential` like buying a pre-built Lego house. It's fast, but you can't change the floor plan.
> The **Functional API** is like buying loose Lego bricks and drawing the blueprints yourself. It requires explicitly defining the foundation (`Input`), but you can build a mansion with multiple wings, bridges, and exits (`Model(inputs=[A, B], outputs=[C, D])`).

![](functional-vs-sequential.svg)

---

## 4.6 Compiling the Model

Once the architecture is stacked, we must **compile** the model. This step configures the training process by binding three crucial components together.

```python
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
```

1. **Optimizer:** Determines *how* the neural network updates its weights. We previously hardcoded standard "Gradient Descent." Modern optimizers like **Adam** (Adaptive Moment Estimation) are far superior, dynamically adjusting learning rates for individual weights to converge much faster.
2. **Loss Function:** Measures how wrong the network is. For standard multi-class problems, we use `sparse_categorical_crossentropy`. For binary problems, `binary_crossentropy`.
3. **Metrics:** Human-readable metrics to judge performance during training. While "Loss" goes down, "Accuracy" goes up, which is much easier for us to interpret!

---

## 4.7 Training and Evaluating (Fitting)

Finally, we feed our data into the model to begin the training loop. This is done via the `.fit()` method.

```python
history = model.fit(X_train, Y_train, epochs=50, batch_size=32)
```

To understand this, we must precisely define two critical terms:

### Epochs vs. Batches
- **Batch:** A subset of your training data. Instead of feeding all 10,000 images into the network at once (which would melt your computer's RAM), we feed them in batches (e.g., 32 images at a time). The model predicts, calculates loss, and updates gradients *after every batch*.
- **Epoch:** One complete pass over the *entire* dataset. If you have 1,000 examples and a batch size of 100, it takes 10 batches (and 10 weight updates) to complete 1 Epoch.

### Monitoring Training
The `history` object returned by `.fit()` contains a record of loss and accuracy values at successive epochs, allowing you to plot the learning curve.

Furthermore, Keras allows you to use **Callbacks**. These are functions executed at the end of specific epochs. 
- For example, **Early Stopping** can be used to automatically halt training the moment the model stops improving, saving time and preventing over-memorization (overfitting).
- **Model Checkpointing** saves the weights of your model dynamically so you never lose a well-trained state if your computer crashes.

By mastering both the `Sequential` and `Functional` APIs, you've unlocked the power to build, scale, and iterate on complex Deep Learning models in minutes rather than days.

---

## ==Appendix: A Quick-Reference Guide to Loss Functions==

Choosing the correct loss function is critical; if you tell the model to optimize the wrong metric, it will learn the wrong thing. Here is a cheat sheet of the 7 most commonly used loss functions in the industry.

### For Regression Problems (Predicting Continuous Numbers)

**1. Mean Squared Error (MSE / L2 Loss)**
- **What it is:** The average of the squared differences between predicted and actual values.
- **Equation:** $L = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$
- **Derivative:** $\frac{\partial L}{\partial \hat{y}_i} = -\frac{2}{N}(y_i - \hat{y}_i)$
- **When to use it:** This is the default loss function for almost all regression problems (predicting house prices, temperature, etc.).
- **Pro-Tip:** Because it *squares* the errors, it punishes large mistakes (outliers) very heavily.

![[MSE-Loss-Curve.png]]

**2. Mean Absolute Error (MAE / L1 Loss)**
- **What it is:** The average of the absolute differences between predicted and actual values.
- **Equation:** $L = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$
- **Derivative:** $\frac{\partial L}{\partial \hat{y}_i} = -sign(y_i - \hat{y}_i)$
- **When to use it:** Use this when your dataset has a lot of extreme **outliers**. Since it doesn't square the error, a massive outlier won't wildly destroy your model's gradient.
![[MAE-Loss-Curve.png]]

**3. Huber Loss**
- **What it is:** A mathematical hybrid between MSE and MAE. It acts like MSE when the error is small, but becomes linear (like MAE) when the error is large.
- **Equation:** 
  $$ L = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases} $$
- **Derivative:** 
  $$ \frac{\partial L}{\partial \hat{y}} = \begin{cases} -(y - \hat{y}) & \text{for } |y - \hat{y}| \le \delta \\ -\delta \cdot sign(y - \hat{y}) & \text{otherwise} \end{cases} $$
- **When to use it:** When you want the smooth convergence of MSE but need the robustness to outliers of MAE.

![[Huber-Loss-Curve.png]]

### For Classification Problems (Predicting Categories)

**4. Binary Cross-Entropy (BCE)**
- **What it is:** Measures the "distance" between two probability distributions for a 0/1 outcome.
- **Equation:** $L = -\frac{1}{N}\sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$
- **Derivative:** $\frac{\partial L}{\partial \hat{y}_i} = -\left( \frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i} \right) = \frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}$
- **When to use it:** Any **Yes/No** or **True/False** problem (e.g., Is this an image of a cat? Yes or No). 
- **Required Activation:** Your final layer *must* use a `sigmoid` activation function to squish outputs between 0 and 1.

![[Binary-Cross-Entropy-Loss-Curve.png]]

**5. Categorical Cross-Entropy (CCE)**
- **What it is:** The expansion of BCE for multiple classes ($C$).
- **Equation:** $L = -\sum_{c=1}^C y_c \log(\hat{y}_c)$
- **Derivative:** $\frac{\partial L}{\partial \hat{y}_c} = -\frac{y_c}{\hat{y}_c}$
- **When to use it:** Multi-class classification tasks where your labels are **One-Hot Encoded** (e.g., predicting an animal: `[0, 1, 0, 0]`).
- **Required Activation:** Your final layer *must* use a `softmax` activation function so all output probabilities sum to 1.

**6. Sparse Categorical Cross-Entropy (SCCE)**
- **What it is:** Mathematically identical to CCE under the hood, but computes probabilities based on a single integer representing the correct class.
- **Equation & Derivative:** Same as Categorical Cross-Entropy! The difference is in *how the data is formatted* rather than how the loss is calculated.
- **When to use it:** Multi-class classification tasks where your labels are **Integers** (e.g., predicting an animal: `2` instead of a one-hot array). 
- **Pro-Tip:** This is highly preferred over standard CCE for datasets with thousands of categories (like vocabulary words in NLP) because it saves massive amounts of memory by avoiding giant sparse matrices.

**7. Kullback-Leibler Divergence (KL Divergence)**
- **What it is:** Measures how much one probability distribution ($Q$) differs from a reference distribution ($P$).
- **Equation:** $D_{KL}(P || Q) = \sum_x P(x) \log\left(\frac{P(x)}{Q(x)}\right)$
- **Derivative (w.r.t model output $Q(x)$):** $\frac{\partial D_{KL}}{\partial Q(x)} = -\frac{P(x)}{Q(x)}$
- **When to use it:** Often used in complex generative architectures like **Variational Autoencoders (VAEs)** to force the network's internal representations to match a standard normal distribution.
