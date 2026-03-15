# 5. Callbacks and TensorBoard in Keras

> [!quote] "Action is the foundational key to all success." — **Pablo Picasso** (and probably your Neural Network if it had a mouth).

In our previous chapters, training a network was like tossing a pizza into a brick oven, locking the door, and just praying it came out perfectly 50 epochs later. We had no visibility into what was happening *inside* the oven while it was baking—we just waited for the smoke or the smell of success.

This chapter introduces the tools professional Data Scientists use to peek inside the oven, monitor the temperature, and ensure the model isn't turning into a burnt cracker. We're moving from "Hope and Pray" to "Monitor and Control."

---

## 5.1 Introduction to Callbacks

A **Callback** is an object containing a set of functions that are executed at specific stages during the training procedure (e.g., at the start of a batch, the end of an epoch, or the end of training).

> [!abstract] **Teacher's Analogy:** 
> Imagine you're teaching a toddler to ride a bike. A callback is like you running alongside them. 
> - If they start drifting toward a rose bush, you gently nudge them back (**LearningRateScheduler**).
> - If they suddenly master it and start doing wheelies, you tell them they can stop for the day (**EarlyStopping**).
> - Every time they complete a lap around the park, you take a polaroid picture to show their progress (**ModelCheckpoint**).
> 
> Without you (the callback), the toddler just keeps pedaling until they hit a wall or run out of juice!

### Verbosity
The simplest, built-in "callback" behavior is controlled by the `verbose` argument in `model.fit()`.
- `verbose=1`: The model prints a beautiful progress bar, loss, and accuracy data after every epoch.
- `verbose=0`: Silent mode. The model prints absolutely nothing while it trains.

---

## 5.2 Creating Custom Callbacks

While Keras provides many built-in callbacks, it is incredibly easy to build your own. You simply create a Python class that inherits from `tf.keras.callbacks.Callback`. 

When you inherit this parent class, you gain access to several powerful "hook" methods that you can override to execute your own custom code.

### The Hierarchy of Override Methods
You can inject your custom code at three different granularities. Think of this as the "Zoom Level" of your monitoring:

| Level | Method | When does it trigger? |
| :--- | :--- | :--- |
| **Global** | `on_train_begin / end` | When the marathon starts and when it finishes. |
| **Epoch** | `on_epoch_begin / end` | Every time the runner completes one full lap of the track. |
| **Batch** | `on_train_batch_begin / end` | Every single step the runner takes (Warning: Very noisy!). |

### Example: A Custom Alert Callback
```python
import tensorflow as tf

class VeryLoudCallback(tf.keras.callbacks.Callback):
    
    # We override the epoch_end method
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2000 == 0:
            print(f"\n🚨 ALERT! Epoch {epoch} completed! Current Loss: {logs['loss']:.4f}")

# To use it, simply pass a LIST of instantiated callbacks to the fit method!
my_loud_callback = VeryLoudCallback()
model.fit(X_train, Y_train, epochs=10000, callbacks=[my_loud_callback])
```

---

## 5.3 Essential Built-in Callbacks

Nobody wants to reinvent the wheel. Keras comes packed with highly optimized callbacks for standard industry tasks. Here is the daily-driver toolkit:

### 1. `CSVLogger`
Saves your loss and metric history directly to a CSV file. Crucial for plotting learning curves later without losing data if your notebook crashes.
```python
csv_logger = tf.keras.callbacks.CSVLogger("training_history.csv")
```

### 2. `EarlyStopping`
The ultimate defense against **overfitting** (when your model memorizes the training data but fails on new data). This callback halts training the moment the model stops improving.
```python
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
```

### 3. `ModelCheckpoint`
Saves the intermediate weights (the $W$ and $b$ matrices) of your model during training. You can configure it to only save the *best* weights so it doesn't overwrite a good model if an epoch goes badly.
```python
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
```

### 4. `LearningRateScheduler`
Changes your learning rate ($\alpha$) dynamically based on a schedule you define (e.g., "halve the LR every 10 epochs").

### 5. `ReduceLROnPlateau` 
> [!tip] **The "Patient Care" Callback**
> If `EarlyStopping` is like a doctor saying "Go home, you're not getting better," then `ReduceLROnPlateau` is like a doctor saying "Let's try a smaller dose of medicine because the current one isn't working anymore."
> 
> It monitors a metric (like `val_loss`) and if it hasn't improved for a few epochs, it automatically shrinks the learning rate to help the model "fine-tune" its weights.

```python
# If val_loss doesn't improve for 5 epochs, multiply LR by 0.1
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
```

### 6. `LambdaCallback`
For when you're feeling lazy and don't want to write a whole class. It lets you create simple callbacks on the fly using anonymous `lambda` functions.

```python
# A quick one-liner to print something at the start of every epoch
simple_print = tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch,logs: print(f"🚀 Launching Epoch {epoch}"))
```

---

## 5.4 TensorBoard: The Ultimate Dashboard

> [!quote] "The advance of science is the discovery of some new instrument of research which allows the expansion of our senses." — **Humphry Davy**

While printing text to a console is fine (if you enjoy reading Matrix-style falling code), humans are visual creatures. **TensorBoard** is the "Mission Control" for your neural network. It's a spectacular web-based dashboard designed to visualize the internal states of your model in real-time.

It allows you to:
- Graph **Metrics** (beautiful, smoothing curves of Loss and Accuracy).
- Visualize the **Model Graph** (seeing how your sequence of layers is physically connected).
- View **Histograms** (watching the actual numerical distribution of your $W$ and $b$ matrices shift over time).
- Display images, text, and audio data flowing through the network.

### Installation and Setup
Ensure it is installed via your package manager:
```bash
pip install tensorboard
# OR
conda install -c conda-forge tensorboard
```

If you are using a Jupyter Notebook or Google Colab, you can load TensorBoard directly into the cell using a magic command:
```python
# Load the extension
%load_ext tensorboard

# Define where your logs will live
log_folder = 'logs/'

# To reload if it freezes
%reload_ext tensorboard
```

### Launching the Dashboard
You spin up the local webserver pointing to your log directory using:
```bash
%tensorboard --logdir={log_folder}
```

---

## 5.5 The `TensorBoard` Callback

To forcefully push data out of Keras and *into* that dashboard, you use the final, most complex built-in callback: the `TensorBoard` callback.

```python
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_folder,
    update_freq='epoch',
    histogram_freq=1,
    write_graph=True,
    write_images=False
)

model.fit(X, Y, callbacks=[tb_callback])
```

### Parameter Deep-Dive:
- `log_dir`: The path where the logs will be saved. **Warning:** Do not share this exact sub-directory with other callbacks, as it corrupts the TensorBoard timeline (it's like trying to record two different movies on the same VHS tape).
- `update_freq`: Determines how often TensorBoard plots a data point. `'epoch'` updates once per full pass. `'batch'` updates furiously fast (which can slow down training—don't be a helicopter parent!).
- `histogram_freq`: Determines how frequently (in epochs) the distribution of weights are computed. Setting it to `1` means visualizing weight shifts every single epoch. 
- `write_graph`: If `True`, draws the physical architecture diagram of your model. Very useful for showing your boss that you actually did work!

---

> [!tip] **Pro Teacher's Tip:**
> Always use `EarlyStopping` and `ModelCheckpoint` together. It's like having a safety net and a save-point in a video game. You won't fall into the pit of overfitting, and if you do, you can always respawn from your best checkpoint!
