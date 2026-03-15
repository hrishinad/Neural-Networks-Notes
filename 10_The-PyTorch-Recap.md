# Chapter 10: The Grand Recap — PyTorch Edition (Keras 3)

In Chapter 9, we built a master workflow using the TensorFlow engine. Now, we perform the same feat using **PyTorch** as our backend. This is the power of Keras 3: the code remains largely the same, but the underlying machinery changes to leverage the PyTorch ecosystem.

---

## 1. The Engine Swap (Configuration)
Before any code runs, we must tell Keras to use the PyTorch engine.

```python
import os
os.environ["KERAS_BACKEND"] = "torch" # Swapping the engine!

import keras
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
```

> [!info] **Object Analogy: The Variable System**
> - **In TensorFlow Backend**: Keras variables are backed by `tf.Variable`.
> - **In PyTorch Backend**: Keras variables are backed by `torch.Tensor` (with `requires_grad=True`).
> - **Relationship**: When you create a layer in Keras-Torch, the weights are internally instantiated as `torch.nn.Parameter`. This makes your Keras model a native **PyTorch `nn.Module`**.

---

## 2. The Data Phase (PyTorch Native)
Instead of `tf.data`, we use the PyTorch **DataLoader**. This is the standard way to feed data in the Torch ecosystem.

```python
# 1. Prepare Data (Standard NumPy)
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Convert to Torch Tensors and Dataset
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), 
    torch.tensor(y_train, dtype=torch.float32)
)

# 3. Create DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
```

---

## 3. Architecture: Sequential vs. Functional
Just like in Chapter 9, Keras 3 allows us to choose between the linear Sequential API and the flexible Functional API.

### Fork: Building with the PyTorch Backend
```python
from keras import layers, Sequential, Model

# --- Sequential Implementation ---
model_seq = Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(50, activation="relu", kernel_initializer="he_normal"),
    layers.BatchNormalization(),
    layers.Dense(1)
])

# --- Functional Implementation ---
input_ = layers.Input(shape=(X_train.shape[1],))
h1 = layers.Dense(50, activation="relu", kernel_initializer="he_normal")(input_)
bn1 = layers.BatchNormalization()(h1)
output = layers.Dense(1)(bn1)

model_func = Model(inputs=[input_], outputs=[output])
```

---

## 4. Automation: Keras Tuner (Backend Agnostic)
Does Keras Tuner change? **No.** The API for defining hypermodels and running searches is identical to Chapter 9.

> [!abstract] **The "Brain" vs. The "Body"**
> Think of **Keras Tuner** as the "Brain" deciding which hyperparameters to try. It doesn't care if the "Body" (the model) is running on TensorFlow or PyTorch. It simply tells Keras what to build, and Keras 3 handles the backend-specific execution.

```python
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    
    # Tuning remains the same!
    model.add(layers.Dense(
        units=hp.Int("units", 32, 128, step=32),
        activation="relu"
    ))
    model.add(layers.Dense(1))
    
    model.compile(optimizer="adam", loss="mse")
    return model

tuner = kt.RandomSearch(build_model, objective="val_loss", max_trials=5)

# You can even pass the PyTorch DataLoader to the tuner!
tuner.search(train_loader, epochs=5)
```

---

## 5. The Compile & Fit Phase
We "wire" the model using Keras optimizers, which act as wrappers for PyTorch optimizers.

```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Apply to either model
model_seq.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
model_func.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# Pass the PyTorch DataLoader directly to fit()
model_seq.fit(train_loader, epochs=10)
```

> [!warning] **The "Logic" Shift: Autograd vs. GradientTape**
> - **TensorFlow (Chapter 9)**: Uses `tf.GradientTape` to "record" operations for differentiation.
> - **PyTorch (Chapter 10)**: Uses **Autograd**. The graph is built dynamically as the forward pass executes.
> - **Result**: On the PyTorch backend, Keras models are more flexible for dynamic logic (like `if/else` conditions inside a layer) because the graph isn't "pre-compiled" in the same way.

---

## 6. Summary Checklist
- [x] **Setup**: Engine swapped via `os.environ["KERAS_BACKEND"] = "torch"`.
- [x] **Data**: Native PyTorch `DataLoader` used for training.
- [x] **Architecture**: Both `Sequential` and `Functional` APIs remain standard.
- [x] **Tuning**: `Keras Tuner` works seamlessly, consuming PyTorch loaders.
- [x] **Training**: `model.fit()` bridges the gap between high-level UI and the PyTorch engine.
