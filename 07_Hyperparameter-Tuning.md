# 7. Hyperparameter Tuning: Mastering the Art of the Recipe

> [!quote] "The difference between a good model and a great model is often not the architecture, but the patience of the person tuning it." — Anonymous ML Engineer

Hyperparameter tuning is the final frontier in building high-performance Neural Networks. While back-propagation handles the learning of weights and biases (parameters), **you** are responsible for setting the environment in which that learning happens. 

Think of it like being a professional chef. The weights are the specific amounts of salt and pepper added during the cooking process, but the hyperparameters are the **recipe** itself: the oven temperature, the choice of flour, and how long the dough was allowed to rest.

---
## 7.1 The "Pizza Oven" Analogy

To understand hyperparameters, let's step into a high-end pizzeria.

> [!abstract]
> - **Ingredients (Data)**: The quality of your tomatoes and flour determines the ceiling of your pizza quality.
> - **Oven Temperature (Learning Rate)**: 
>     - Too hot (High LR): The crust burns before the cheese melts (Model diverges).
>     - Too cold (Low LR): The pizza takes forever to cook and ends up soggy (Training is too slow).
> - **Cooking Time (Epochs)**: 
>     - Too short: Raw dough (Underfitting).
>     - Too long: Charcoal (Overfitting).
> - **Dough Thickness (Model Capacity)**:
>     - Too thin: Can't hold the toppings (Model is too simple for the data).
>     - Too thick: You can't taste the sauce (Model is too complex, captures noise).
> - **Toppings (Dropout/Regularization)**: Adding just enough variety to keep things interesting without overwhelming the palate.

---

## 7.2 Key Hyperparameters in Neural Networks

Broadly, we categorize hyperparameters into three buckets:

### 1. Architectural Hyperparameters
These define the "skeleton" of your model.
- **Number of Layers**: Depth of the network. More layers allow for more abstract features but increase the risk of vanishing gradients.
- **Hidden Units ($n_h$)**: The width of each layer. 
- **Activation Functions**: ReLU, Tanh, etc. (Though usually fixed after experimentation).

### 2. Optimization Hyperparameters
These define how the model "moves" during training.
- **Learning Rate ($\alpha$)**: The most critical hyperparameter. It controls the step size at each iteration.
- **Batch Size ($B$)**: How many samples to look at before updating weights.
- **Optimizer Choice**: Adam, RMSprop, SGD with Momentum.
- **Learning Rate Schedule**: Decaying $\alpha$ over time.

### 3. Regularization Hyperparameters
These prevent the model from "memorizing" the training data.
- **Dropout Rate**: The probability of "turning off" a neuron during a training pass.
- **Lambda ($\lambda$)**: The strength of L1/L2 weight penalties.

---

## 7.3 Regularization: Fighting the "Memorization" Monster

When a model has too many parameters relative to the amount of data, it starts memorizing the noise instead of learning the signal. This is **Overfitting** (High Variance). Regularization is the process of adding a "penalty" to the loss function to keep weights small.

### 7.3.1 The Math of Regularization
The standard cost function $J(W, b)$ is modified by adding a regularization term:

$$J_{regularized} = J(W, b) + \frac{\lambda}{2m} \Omega(W)$$

Where:
- $\lambda$: Regularization parameter (Hyperparameter).
- $m$: Number of training examples.
- $\Omega(W)$: The penalty term.

### 7.3.2 L2 Regularization (Weight Decay)
In L2 regularization, we use the squared Euclidean norm of the weights:
$$\Omega(W) = \|W\|_2^2 = \sum w_j^2$$

- **Intuition**: L2 penalizes large weights heavily (squaring them). This forces the weights to spread out across all features rather than relying on just one.
- **Result**: Weights become small but rarely exactly zero. 
- **Visual**: Think of it as a "circular" constraint. It pulls the weights towards the origin in a smooth, balanced way.

```python title:"Implementing L2 in Keras"
from tensorflow.keras import layers, regularizers

model.add(layers.Dense(64, activation='relu', 
                       kernel_regularizer=regularizers.l2(0.01))) # 0.01 is lambda
```

### 7.3.3 L1 Regularization (Sparsity)
In L1 regularization, we use the absolute values of the weights:
$$\Omega(W) = \|W\|_1 = \sum |w_j|$$

- **Intuition**: L1 penalizes weights linearly. Because the derivative of $|w|$ is a constant ($\pm 1$), it keeps pushing weights all the way to zero.
- **Result**: It performs **Feature Selection**. Many weights become exactly $0$, giving you a "sparse" model.
- **Visual**: A "diamond" constraint. Costs contours hit the "corners" of the diamond on the axes, resulting in zeros.

![](https://miro.medium.com/v2/resize:fit:1600/1*_e8BLNA749W_7yxi7hz-DA.gif)

```python title:"Implementing L1 in Keras"
model.add(layers.Dense(64, activation='relu', 
                       kernel_regularizer=regularizers.l1(0.01)))
```

---

## 7.4 Dropout: The "Random Bench" Strategy

Dropout is a specialized regularization technique for Neural Networks. At each training step, we randomly "drop out" (deactivate) a fraction of neurons in a layer.

### 7.4.1 Why it works?
> [!abstract] The Ensemble Analogy
> Imagine a company where every employee is $100\%$ reliable. They might get lazy and rely on one "superstar" coder. If that superstar leaves, the company fails. 
> Dropout is like randomly forcing employees to take a day off. This forces **everyone** to learn the codebase and become competent. No one can rely on a single neuron to do all the work.

### 7.4.2 Mechanism
1. For each training iteration, generate a mask (a matrix of $0$s and $1$s) from a Bernoulli distribution with probability $p$.
2. Multiply the activations by this mask.
3. During **Inference** (Testing), we turn off Dropout and scale the weights by $p$ to maintain the expected value of the output.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8127972%2F334849d2c3a55abf89ad1a99cb78fe7c%2FDroupout.gif?generation=1693075894689828&alt=media)

```python title:"Implementing Dropout in Keras"
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5)) # 50% of neurons will be deactivated randomly
```

---

## 7.5 Search Techniques: How to find the "Perfect" Recipe

How do we actually find the best values? We don't just guess. We use search strategies.

### 1. Grid Search (The Perfectionist)
You define a set of values for each hyperparameter and try **every possible combination**. Very slow and prone to the "curse of dimensionality."

### 2. Random Search (The Lucky Explorer)
Picks random combinations within a range. Empirically, it is **better** than Grid Search because it explores the search space more effectively.

### 3. Bayesian Optimization (The Smart Student)
Uses past results to build a probabilistic model of the loss surface and chooses the next point to minimize expected loss (Exploration vs. Exploitation).

![](https://images.contentstack.io/v3/assets/bltb654d1b96a72ddc4/blt500831b25ec72372/660f40c8e838c8586360f703/SPC-Blog-Hyperparameter-optimization-2.jpg)

---

## 7.6 The Order of Operations

In what order should you tune? While it depends on the project, the "Golden Order" is:
1. **Learning Rate ($\alpha$)**: By far the most important.
2. **Optimizer & Batch Size**: Get the training stable.
3. **Number of Units & Layers**: Fine-tune the capacity.
4. **Dropout & Regularization**: Polish the generalization.

---

## 7.7 Troubleshooting Performance: The Tuning Playbook

| Scenario                          | What happened?                                                             | What to Tweak?                                                                                                                                  |
| :-------------------------------- | :------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bad Training Performance**      | Underfitting (High Bias). The model isn't even learning the training data. | 1. Increase Model Capacity (Layers/Units). <br> 2. Decrease $\lambda$ (Regularization). <br> 3. Tune Learning Rate. <br> 4. Change Weight Init. |
| **Good Training, Bad Validation** | Overfitting (High Variance). The model is "memorizing" noise.              | 1. Increase Dropout Rate. <br> 2. Increase $\lambda$ (L2). <br> 3. Early Stopping. <br> 4. More Data.                                           |
| **Good Train & Val, Bad Test**    | Distribution Shift or Data Leakage.                                        | 1. Check data splits. <br> 2. Verify if test set matches the training distribution.                                                             |

![](https://www.mathworks.com/discovery/overfitting/_jcr_content/mainParsys/image.adapt.full.medium.svg/1767870489297.svg)

---

## 7.8 Full Implementation: Keras Tuner

Here is how you combine everything into an automated search:

```python title:"Implementation: Keras Tuner"
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_model(hp):
    # Using the Functional API
    inputs = keras.Input(shape=(28, 28)) # Explicitly defining the foundation
    x = layers.Flatten()(inputs)
    
    # Tuning Hidden Layers & Units
    for i in range(hp.Int('num_layers', 1, 3)):
        x = layers.Dense(
            units=hp.Int(f'units_{i}', 32, 256, step=32),
            activation='relu',
            # Tuning L2 Regularization strength
            kernel_regularizer=regularizers.l2(hp.Float('l2', 1e-4, 1e-2, sampling='log'))
        )(x)
        # Adding Dropout after each dense layer
        x = layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1))(x)
    
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Tuning Optimizer & Learning Rate
    hp_lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='intro_to_kt'
)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

---

## Summary
Hyperparameter tuning is the bridge between a theoretical architecture and a production-grade model. Regularization (L1/L2) and Dropout are your main tools to prevent the model from getting "too smart for its own good." Next time, we'll dive into **Mini-batch Gradient Descent**, **Batch Normalization** and **Autoencoders**.
