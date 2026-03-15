# Chapter 0: The Journey Begins — Introduction & Roadmap

Welcome to the **Notes on Neural Networks and Deep Learning**! 🚀

Neural Networks are the engine room of modern AI—from the recommendation systems that suggest your next favorite show to the generative models that create art and code. This course is designed to take you from "What is a neuron?" to building, tuning, and deploying production-grade deep learning models.

> [!abstract] Course Philosophy: Analogy-Math-Code
> We believe that deep learning is best learned through a three-step workflow:
> 1. **Analogy**: Understanding the core concept with a relatable, real-world example.
> 2. **Math**: Deriving the rigorous formulas and matrix operations that make it work.
> 3. **Code**: Implementing the solution using industry-standard tools like NumPy, Keras, and PyTorch.

---

## 🗺️ Course Roadmap (Table of Contents)

Below is the high-level overview of our 10-chapter journey. Each chapter builds upon the previous one, layering complexity until we reach the "Grand Recap."

- [b] **[[01_Introduction-to-Neural-Networks|Chapter 1: The AI Landscape]]**
	- The evolution: AI vs. ML vs. DL.
	- The role of Feature Engineering and why Deep Learning changed the game.
	- High-level use cases for FFNN, CNN, and RNN.
	- Understanding the "Black Box" and the challenge of Overfitting.

- [b] **[[02_Forward-and-Backward-Propagation|Chapter 2: The Biological Engine]]**
	- Anatomy of a Neuron: $Z = WX + b$.
	- Activation Functions: Sigmoid, Tanh, ReLU, and Softmax.
	- The core of learning: Forward and Backward Propagation.
	- Dealing with Vanishing and Exploding Gradients.
	- Pure NumPy implementation of a single neuron.

- [b] **[[03_Neural-Network-with-2-Hidden-Layers|Chapter 3: Deep Architectures]]**
	- Moving from a single neuron to multi-layer networks.
	- Mathematical derivation of a 2-Hidden Layer architecture.
	- Managing matrix dimensions like a pro.
	- Using the Pythonic `@` operator and dictionary-based parameter storage.

- [b] **[[04_Introduction-to-TensorFlow-and-Keras|Chapter 4: The Framework Era (Keras)]]**
	- Understanding the stack: TensorFlow (Engine) vs. Keras (Dashboard).
	- Building models with the **Sequential API** and the **Functional API**.
	- The lifecycle: Compile, Fit, Evaluate, and Predict.
	- Appendix of Loss Functions: MSE, MAE, BCE, and CCE.

- [b] **[[05_Callbacks-and-TensorBoard|Chapter 5: Visibility & Control]]**
	- Gaining insight during training with **TensorBoard**.
	- Automating model improvement with **Callbacks**.
	- Built-ins: `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`.
	- Writing custom callbacks for specialized logic.

- [b] **[[06_Optimizers-and-Weight-Initialization|Chapter 6: Optimization & Stability]]**
	- The importance of "Symmetry Breaking."
	- Weight Initialization strategies: Xavier (Glorot) and He.
	- Beyond SGD: Momentum, RMSprop, and the **Adam Optimizer**.

- [b] **[[07_Hyperparameter-Tuning|Chapter 7: The Art of Tuning]]**
	- Parameters vs. Hyperparameters.
	- Search strategies: Grid Search, Random Search, and Bayesian Optimization.
	- Mastering the tuning order and troubleshooting performance.
	- Hands-on with **Keras Tuner**.

- [b] **[[08_Batch-Normalization-and-Autoencoders|Chapter 8: Normalization & Feature Learning]]**
	- Improving training stability with **Batch Normalization**.
	- Understanding Mini-batch Gradient Descent.
	- Intro to **Autoencoders**: Data compression and denoising architectures.

- [b] **[[09_The-Grand-Recap|Chapter 9: The Keras Masterclass]]**
	- An end-to-end case study integrating everything.
	- Data scaling, architectural choices, and hyperparameter tuning in one project.

- [b] **[[10_The-PyTorch-Recap|Chapter 10: The PyTorch Transition]]**
	- Implementing the "Master Model" in PyTorch.
	- Working with Tensors, DataLoaders, and `nn.Module`.
	- Manual training loops and logging with Optuna.

- [b] **[[11_Model-Explainability|Chapter 11: Model Explainability (XAI)]]**
	- Peeking into the "Black Box."
	- Global vs. Local Feature Importance.
	- Inherently Interpretable Models (Trees).
	- LIME: The Local Surrogate Model approach.
	- Practical LIME + Keras case study.

---

> [!tip] How to use these notes
> If you are a beginner, start from **Chapter 1**. If you already know the basics of propagation, you might want to jump to **Chapter 4** to start using Keras, or **Chapter 10** if you are specifically looking to transition to PyTorch.

Let's dive in! 🧠💻
