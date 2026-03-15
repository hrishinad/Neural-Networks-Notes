# 3. Neural Network with 2 Hidden Layers

Welcome to Chapter 3! Now that we have covered the foundational concepts of a simple neural network and the rigorous mathematics governing its training, it's time to go **deeper**. 

In the real world, complex problems (like recognizing a face in an image or translating languages) require networks capable of learning highly non-linear, abstract representations. A network with just one hidden layer often isn't enough. We need multiple layers to hierarchicalize learning.

> [!abstract] **Teacher's Analogy:** 
> Think of a deep neural network like a massive car factory assembly line. 
> - **Layer 1** workers just cut raw metal into basic shapes (edges and lines).
> - **Layer 2** workers weld those basic shapes into doors and hoods (complex shapes).
> - **Layer 3** workers assemble the doors and hoods into a recognizable car (the final object). 
> Each layer aggressively builds upon the abstractions of the previous one!

Let's explore the mathematics and implementation of a **2-Hidden Layer Neural Network**.

---

## 3.1 Network Architecture

Let's define our new depth. We will build a classification network with the following structure:

- **Input Layer ($X$):** 2 features ($x_1, x_2$)
- **Hidden Layer 1 ($l=1$):** 2 Neurons, **ReLU** Activation ($A^{[1]}$)
- **Hidden Layer 2 ($l=2$):** 2 Neurons, **ReLU** Activation ($A^{[2]}$)
- **Output Layer ($l=3$):** 1 Neuron, **Softmax** Activation ($A^{[3]}$) representing the final prediction ($\hat{y}$) for our classes.

![](2-Layer-Neural-Network.svg)

### The New Weight Matrices

> [!warning] **The Programmer's Nightmare:** 
> Adding layers means adding more matrices. If you thought matching the dimensions of $W^{[1]}$ and $X$ was fun, welcome to the big leagues. If your inner matrix dimensions don't match, Python will throw a `ValueError` so loud your ancestors will hear it. Always check your shapes!

Adding a layer means adding a whole new set of weights and biases to keep track of. Let's look at the dimensions:

- $W^{[1]}$: Weights connecting Input ($2$) to Hidden 1 ($2$). Shape: $(2, 2)$
- $b^{[1]}$: Biases for Hidden 1. Shape: $(2, 1)$
- $W^{[2]}$: Weights connecting Hidden 1 ($2$) to Hidden 2 ($2$). Shape: $(2, 2)$
- $b^{[2]}$: Biases for Hidden 2. Shape: $(2, 1)$
- $W^{[3]}$: Weights connecting Hidden 2 ($2$) to Output ($1$). Shape: $(1, 2)$
- $b^{[3]}$: Bias for Output. Shape: $(1, 1)$

---

## 3.2 The Forward Pass

Forward propagation is simply stepping through the network sequentially. calculate the pre-activation ($Z$), apply the activation ($A$), and feed it to the next layer. 

For $m$ training examples, our vectorized equations are:

### Layer 1 (Input $\rightarrow$ Hidden 1)
1. **Linear Transformation:**  
   $$ Z^{[1]} = W^{[1]} \cdot X + b^{[1]} $$
2. **Activation (ReLU):**  
   $$ A^{[1]} = \max(0, Z^{[1]}) $$

### Layer 2 (Hidden 1 $\rightarrow$ Hidden 2)
1. **Linear Transformation:** Notice $A^{[1]}$ is now the "input"!
   $$ Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]} $$
2. **Activation (ReLU):**  
   $$ A^{[2]} = \max(0, Z^{[2]}) $$

### Layer 3 (Hidden 2 $\rightarrow$ Output)
1. **Linear Transformation:** 
   $$ Z^{[3]} = W^{[3]} \cdot A^{[2]} + b^{[3]} $$
2. **Activation (Softmax):** 
   $$ A^{[3]} = \text{Softmax}(Z^{[3]}) = \hat{Y} $$

---

## 3.3 Loss Calculation

Since we are doing classification, we continue to use **Categorical Cross-Entropy Loss**. 

For a single example:
$$ \mathcal{L}(\hat{y}, y) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i) $$

For $m$ training examples (vectorized):
$$ J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} Y_{c}^{(i)} \log(A^{[3] (i)}_{c}) $$

---

## 3.4 The Backward Pass (Extended Chain Rule)

Here is where the magic (and the math) happens. We now have to trace the error $\partial J$ all the way back from Layer 3, through Layer 2, down to Layer 1. The chain rule gets longer!

Let's derive the gradients step-by-step.

### Step 1: Gradients at the Output Layer ($l=3$)

Just like in Chapter 2, the derivative of Cross-Entropy Loss combined with Softmax is incredibly elegant:

$$ dZ^{[3]} = A^{[3]} - Y $$

Now we calculate the gradients for the weights and biases of Layer 3:
$$ dW^{[3]} = \frac{1}{m} dZ^{[3]} \cdot (A^{[2]})^T $$
$$ db^{[3]} = \frac{1}{m} \sum dZ^{[3]} $$

### Step 2: Gradients at Hidden Layer 2 ($l=2$)

To find the error at Layer 2, we must pass the error from $dZ^{[3]}$ backward through the weights $W^{[3]}$, and then multiply by the derivative of the ReLU activation at Layer 2.

1. **Calculate error prior to activation:**
   $$ dA^{[2]} = (W^{[3]})^T \cdot dZ^{[3]} $$
2. **Distribute error through ReLU:**
   $$ dZ^{[2]} = dA^{[2]} * g'^{[2]}(Z^{[2]}) $$
   *(Remember: $*$ is element-wise multiplication, and $g'$ is the derivative of ReLU, which is 1 if $Z > 0$, else 0).*

Now compute the Gradients for Layer 2:
$$ dW^{[2]} = \frac{1}{m} dZ^{[2]} \cdot (A^{[1]})^T $$
$$ db^{[2]} = \frac{1}{m} \sum dZ^{[2]} $$

### Step 3: Gradients at Hidden Layer 1 ($l=1$)

We apply the exact same chain rule logic, grabbing the error from Layer 2 ($dZ^{[2]}$) and passing it backwards!

1. **Calculate error prior to activation:**
   $$ dA^{[1]} = (W^{[2]})^T \cdot dZ^{[2]} $$
2. **Distribute error through ReLU:**
   $$ dZ^{[1]} = dA^{[1]} * g'^{[1]}(Z^{[1]}) $$

Now compute the Gradients for Layer 1:
$$ dW^{[1]} = \frac{1}{m} dZ^{[1]} \cdot X^T $$
$$ db^{[1]} = \frac{1}{m} \sum dZ^{[1]} $$

> [!TIP] The Pattern of Backpropagation
> Do you notice the pattern? For any hidden layer $l$:
> 1. $dZ^{[l]} = (W^{[l+1]})^T \cdot dZ^{[l+1]} * g'^{[l]}(Z^{[l]})$
> 2. $dW^{[l]} = \frac{1}{m} dZ^{[l]} \cdot (A^{[l-1]})^T$
> 3. $db^{[l]} = \frac{1}{m} \sum dZ^{[l]}$
> 
> This elegant, repetitive formula is why neural networks can be easily scaled to 10, 50, or 100 layers in code using loops!

---

## 3.5 Real-World "Pythonic" Implementation

In Chapter 2, our code was a simple script. Here, we will make use of modern Python features (like the `@` matrix multiplication operator) and cleaner data structures (like dictionaries to hold our parameters) to make the code highly legible.

Our goal is to write a procedural script where the Python code looks *almost identical* to the math equations we derived above!

```python
import numpy as np

# --- 1. Activation Functions ---
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    # Shift Z for numerical stability
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# --- 2. Architecture & Data Setup ---
# 2 Features, 3 Examples
X = np.array([[0.1, 0.5, 0.9],
              [0.2, 0.4, 0.1]]) 
m = X.shape[1] # Number of examples (3)

# 3 Classes (One-Hot Encoded target)
Y = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Layer dimensions: [Input(2), Hidden1(2), Hidden2(2), Output(3)]
dims = [2, 2, 2, 3]

# --- 3. Parameter Initialization (He Initialization) ---
# Using Python dictionaries to cleanly hold matrices
params = {
    'W1': np.random.randn(dims[1], dims[0]) * np.sqrt(2. / dims[0]),
    'b1': np.zeros((dims[1], 1)),
    'W2': np.random.randn(dims[2], dims[1]) * np.sqrt(2. / dims[1]),
    'b2': np.zeros((dims[2], 1)),
    'W3': np.random.randn(dims[3], dims[2]) * np.sqrt(2. / dims[2]),
    'b3': np.zeros((dims[3], 1))
}

# --- 4. Training Loop ---
learning_rate = 0.05
epochs = 500

for i in range(epochs):
    # --- FORWARD PROPAGATION ---
    # Layer 1
    Z1 = params['W1'] @ X + params['b1']   # pythonic matrix multiplication (@)!
    A1 = relu(Z1)
    
    # Layer 2
    Z2 = params['W2'] @ A1 + params['b2']
    A2 = relu(Z2)
    
    # Layer 3 (Output)
    Z3 = params['W3'] @ A2 + params['b3']
    A3 = softmax(Z3)

    # --- LOSS CALCULATION ---
    # Categorical Cross-Entropy (with 1e-15 to prevent log(0))
    loss = -np.sum(Y * np.log(A3 + 1e-15)) / m
    if i % 100 == 0:
        print(f"Epoch {i} | Loss: {loss:.4f}")

    # --- BACKWARD PROPAGATION ---
    # Step 1: Output Layer (Softmax)
    dZ3 = A3 - Y
    dW3 = (dZ3 @ A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    
    # Step 2: Hidden Layer 2
    dA2 = params['W3'].T @ dZ3
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    # Step 3: Hidden Layer 1
    dA1 = params['W2'].T @ dZ2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    # --- GRADIENT DESCENT UPDATE ---
    params['W1'] -= learning_rate * dW1
    params['b1'] -= learning_rate * db1
    params['W2'] -= learning_rate * dW2
    params['b2'] -= learning_rate * db2
    params['W3'] -= learning_rate * dW3
    params['b3'] -= learning_rate * db3

print("Training Complete!")
```

### Why is this code better?
- **Procedural but Clean:** We removed the complex `DeepNeuralNetwork` class and loops over arrays of layer configurations. It reads exactly like the mathematical formulas derive earlier.
- **Pythonic Matrix Math (`@` Operator):** Notice the use of `params['W1'] @ X` instead of `np.dot()`. In modern Python (PEP 465), the `@` operator performs matrix multiplication, which drastically cuts down on verbosity and makes code look almost identical to the handwritten math!
- **Data Structures (`dicts`):** We packed all weights and biases into a beautifully simple `params` dictionary instead of writing a huge list of separate variables.

That wraps up Chapter 3! We've successfully extended our math and code to handle deep, multi-layered architectures.
