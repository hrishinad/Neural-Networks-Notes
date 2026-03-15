# Chapter 09: The Grand Recap — Building the "Master Model"

In the previous eight chapters, we have dismantled the black box of Neural Networks piece by piece—from the raw math of a single neuron to the sophisticated architectures of Autoencoders and the automation of Keras Tuner. 

This chapter serves as a **Grand Recap**. We will walk through a complete, end-to-end case study using Keras, stitching together every concept we’ve learned into a logical, production-ready pipeline.

---

## 1. The Setup: Data Preparation
Every neural network is only as good as the data it consumes. We begin with the foundational steps: splitting and scaling.

> [!abstract] Analogy: The Chef's Mise en Place
> Before a chef starts cooking (training), they must wash, chop, and organize their ingredients (data). If the ingredients are in wildly different sizes (unscaled), some will overcook while others remain raw.

### Step 1.1: Train-Test Split & Scaling
We use `StandardScaler` to ensure our features have a mean of 0 and a variance of 1, preventing certain features from "dominating" the weight updates simply because of their scale.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# 1. Load Dataset
data = fetch_california_housing()
X, y = data.data, data.target

# 2. Split (CH 1: Overfitting prevention)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2)

# 3. Scale (CH 2: Gradient Flow)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
```

---

## 2. Architecture: Basic Building Blocks
In **Chapter 4**, we learned two ways to build models.

- [i] **Sequential API**: Best for simple stacks where each layer has exactly one input and one output.
- [i] **Functional API**: Necessary for complex graphs (multi-input, multi-output, or residual connections).

### Fork: Sequential vs. Functional
```python
import tensorflow as tf
from tensorflow import keras
from keras import layers

# --- Sequential ---
model_seq = keras.Sequential([
    layers.Input(shape=X_train.shape[1:]),
    layers.Dense(30, activation="relu"),
    layers.Dense(1)
])

# --- Functional ---
input_ = layers.Input(shape=X_train.shape[1:])
hidden = layers.Dense(30, activation="relu")(input_)
output = layers.Dense(1)(hidden)

model_func = keras.Model(inputs=[input_], outputs=[output])
```

---

## 3. Stability: Weight Initialization & Batch Norm
Training deep networks is risky. Weights can explode or vanish.

> [!warning] Symmetry Breaking
> If all weights start at zero, every neuron in a layer will learn the exact same thing. We use **He Initialization** (for ReLU) or **Xavier** (for Sigmoid/Tanh) to break this symmetry.

### Step 3.1: Adding Robustness
We integrate **Batch Normalization** (CH 8) to normalize activations between layers, making the model less sensitive to initialization.

```python
# --- Sequential Implementation ---
model_seq = keras.Sequential([
    layers.Input(shape=X_train.shape[1:]),
    layers.Dense(50, activation="relu", kernel_initializer="he_normal"),
    layers.BatchNormalization(),
    layers.Dense(50, activation="relu", kernel_initializer="he_normal"),
    layers.BatchNormalization(),
    layers.Dense(1)
])

# --- Functional Implementation ---
input_ = layers.Input(shape=X_train.shape[1:])
h1 = layers.Dense(50, activation="relu", kernel_initializer="he_normal")(input_)
bn1 = layers.BatchNormalization()(h1)
h2 = layers.Dense(50, activation="relu", kernel_initializer="he_normal")(bn1)
bn2 = layers.BatchNormalization()(h2)
output = layers.Dense(1)(bn2)

model_func = keras.Model(inputs=[input_], outputs=[output])
```

---

## 4. Optimization: The Engine Room
How does the model actually "learn"? By minimizing the **Loss Function** using an **Optimizer**.

- [*] **Adam (CH 6)**: The industry standard. It combines **Momentum** (accelerating in the right direction) and **RMSprop** (adjusting learning rates per parameter).

### Step 4.1: Compilation
Whether the model is Sequential or Functional, the compilation step remains consistent.

```python
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,    # Momentum decay
    beta_2=0.999   # Scaling decay
)

# Apply to either model
model_seq.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
model_func.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
```

---

## 5. Automation: Hyperparameter Tuning
In **Chapter 7**, we stopped guessing and started searching. **Keras Tuner** automates the trial-and-error process.

### Step 5.1: Tuning the Architecture
In the real world, you typically decide on your API first (Sequential for simplicity, Functional for complexity) and then tune within that framework.

```python
import keras_tuner as kt

# --- Tuner 1: Sequential ---
def build_sequential_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=X_train.shape[1:]))
    for i in range(hp.Int("n_layers", 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f"units_{i}", 32, 128, step=32),
            activation="relu"
        ))
    model.add(layers.Dense(1))
    
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

tuner_seq = kt.RandomSearch(
    build_sequential_model, 
    objective="val_loss", 
    max_trials=5, 
    project_name="seq_tuning"
)

# --- Tuner 2: Functional ---
def build_functional_model(hp):
    input_ = layers.Input(shape=X_train.shape[1:])
    x = input_
    for i in range(hp.Int("n_layers", 1, 3)):
        x = layers.Dense(
            units=hp.Int(f"units_{i}", 32, 128, step=32),
            activation="relu"
        )(x)
    output = layers.Dense(1)(x)
    
    model = keras.Model(inputs=[input_], outputs=[output])
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

tuner_func = kt.RandomSearch(
    build_functional_model, 
    objective="val_loss", 
    max_trials=5, 
    project_name="func_tuning"
)
```

---

## 6. Visibility: Callbacks & TensorBoard
Don't fly blind. Callbacks (CH 5) allow the model to interact with the training process.

```python
import os
run_logdir = os.path.join(os.curdir, "my_logs")

callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
    keras.callbacks.TensorBoard(run_logdir)
]

# Training either model
history = model_func.fit(
    X_train, y_train, 
    epochs=100, 
    validation_data=(X_valid, y_valid),
    callbacks=callbacks
)
```

---

## 7. Advanced: The Autoencoder Recap
Finally, we recall **Chapter 8**. Autoencoders are for **representation learning**.

### Step 7.1: Bottlenecks in Dual Styles
```python
# --- Sequential Autoencoder ---
encoder_seq = keras.Sequential([
    layers.Input(shape=X_train.shape[1:]),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu") # Latent Space
])

# --- Functional Autoencoder ---
input_ = layers.Input(shape=X_train.shape[1:])
e1 = layers.Dense(16, activation="relu")(input_)
latent = layers.Dense(8, activation="relu")(e1)

encoder_func = keras.Model(inputs=[input_], outputs=[latent])

# Reconstructing original dimensions
decoder = layers.Dense(X_train.shape[1])
output = decoder(latent)
autoencoder_func = keras.Model(inputs=[input_], outputs=[output])
```

---

## 8. Summary Checklist
- [+] **Data**: Always scale and split.
- [+] **Architecture**: Fork your logic between Sequential (simple) and Functional (flexible).
- [+] **Stability**: Use He/Xavier initialization and Batch Norm in both styles.
- [+] **Optimizers**: Adam with tuned $\beta$ values for precision.
- [+] **Tuning**: Keras Tuner can bridge both APIs in a single search.
- [+] **Monitoring**: Always use EarlyStopping and TensorBoard.
