<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Deep%20Learning&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=From%20Neurons%20to%20Intelligence%20%E2%80%94%20with%20Mathematics&descAlignY=60&descSize=18" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=22&pause=1000&color=00D4FF&center=true&vCenter=true&width=750&lines=🧠+Perceptron+→+ANN+→+CNN+→+RNN+→+LSTM;📐+Theory+%2B+Math+%2B+Implementation;🔥+Deep+Learning+from+First+Principles;🚀+Built+with+Python+%7C+TensorFlow+%7C+Keras" alt="Typing SVG" />

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![GitHub commits](https://img.shields.io/badge/Commits-23-brightgreen?style=flat-square&logo=github)](https://github.com/Suman18-bit/Deep-Learning/commits)
[![GitHub stars](https://img.shields.io/github/stars/Suman18-bit/Deep-Learning?style=flat-square&logo=github)](https://github.com/Suman18-bit/Deep-Learning/stargazers)
[![Maintained](https://img.shields.io/badge/Maintained-Yes-00C853?style=flat-square)](https://github.com/Suman18-bit/Deep-Learning)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

</div>

---

## 🌟 About This Repository

> **A structured, math-first journey through Deep Learning** — every architecture explained with the underlying theory, equations, and working implementations in Jupyter Notebooks.

This repository covers **five core neural network architectures** with clean notebooks, real datasets, and full mathematical foundations. Follow the learning path in order for maximum understanding.

```
INPUT → [PERCEPTRON] → [ANN] → [CNN] → [RNN] → [LSTM] → INTELLIGENCE ✨
```

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [🗂️ Repository Structure](#️-repository-structure) |
| 2 | [🔵 Perceptron — Single Neuron](#-perceptron--single-neuron) |
| 3 | [🟡 ANN — Artificial Neural Networks](#-ann--artificial-neural-networks) |
| 4 | [🟢 CNN — Convolutional Neural Networks](#-cnn--convolutional-neural-networks) |
| 5 | [🔴 RNN — Recurrent Neural Networks](#-rnn--recurrent-neural-networks) |
| 6 | [🟣 LSTM — Long Short-Term Memory](#-lstm--long-short-term-memory) |
| 7 | [🛠️ Tech Stack](#️-tech-stack) |
| 8 | [🚀 Getting Started](#-getting-started) |
| 9 | [🗺️ Learning Path](#️-learning-path) |

---

## 🗂️ Repository Structure

```
📦 Deep-Learning/
│
├── 📁 PERCEPTRON/
│   ├── 📓 Perceptron.ipynb
│   └── 📊 Single Layer Perceptron Dataset.csv
│
├── 📁 ANN/
│   ├── 📓 ANN_Basics.ipynb
│   ├── 📓 ANN_with_Data_Visualization.ipynb
│   ├── 📓 Autoencoder&Autodecoder.ipynb
│   └── 📓 Normalizing_tensors.ipynb
│
├── 📁 CNN/
│   ├── 📁 Image Classification/
│   ├── 📓 CNN_Basic.ipynb
│   └── 🗜️  DATASET.zip
│
├── 📁 RNN/
│   ├── 📓 Basic_RNN.ipynb
│   └── 📓 Spam_Email_Dectechtion.ipynb
│
├── 📁 LSTM/
│   ├── 📓 LSTM.ipynb
│   └── 📓 Fake/Real_news_Classification.ipynb.ipynb
│
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🔵 Perceptron — Single Neuron

<img src="https://img.shields.io/badge/Level-Beginner-00C853?style=flat-square"/> <img src="https://img.shields.io/badge/Type-Binary%20Classifier-blue?style=flat-square"/>

> *The simplest neural unit — a mathematical model of a biological neuron.*

### 📁 Files

| File | Description |
|------|-------------|
| `Perceptron.ipynb` | Single-layer perceptron from scratch with manual weight updates |
| `Single Layer Perceptron Dataset.csv` | Custom binary classification dataset |

---

### 📐 Mathematical Foundation

#### 🔹 1. Weighted Sum (Net Input)

Each input $x_i$ is multiplied by its weight $w_i$ and summed with bias $b$:

$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

#### 🔹 2. Activation Function — Step Function

$$f(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

#### 🔹 3. Prediction

$$\hat{y} = f(\mathbf{w}^T \mathbf{x} + b)$$

#### 🔹 4. Perceptron Learning Rule (Weight Update)

After each misclassified sample, weights are updated:

$$w_i \leftarrow w_i + \eta \cdot (y - \hat{y}) \cdot x_i$$

$$b \leftarrow b + \eta \cdot (y - \hat{y})$$

Where $\eta$ is the **learning rate**, $y$ is the true label, and $\hat{y}$ is the predicted label.

#### 🔹 5. Decision Boundary

The hyperplane separating the two classes:

$$\mathbf{w}^T \mathbf{x} + b = 0 \implies w_1 x_1 + w_2 x_2 + b = 0$$

#### 🔹 6. Perceptron Convergence Theorem

If the data is **linearly separable**, the perceptron is **guaranteed to converge** in a finite number of steps.

$$\text{Error} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i \neq y_i] \xrightarrow{} 0$$

---

**🔑 Key Concepts:**
`Weighted Sum` · `Step Activation` · `Linear Separability` · `Weight Update Rule` · `Decision Boundary` · `Convergence Theorem`

---

## 🟡 ANN — Artificial Neural Networks

<img src="https://img.shields.io/badge/Level-Intermediate-FFA726?style=flat-square"/> <img src="https://img.shields.io/badge/Type-Multi--Layer%20Network-orange?style=flat-square"/>

> *Stacked layers of neurons learning complex, non-linear representations via backpropagation.*

### 📁 Files

| File | Description |
|------|-------------|
| `ANN_Basics.ipynb` | Layers, activations, forward & backward pass |
| `ANN_with_Data_Visualization.ipynb` | Loss curves, accuracy plots, decision boundaries |
| `Autoencoder&Autodecoder.ipynb` | Unsupervised dimensionality reduction |
| `Normalizing_tensors.ipynb` | Tensor preprocessing techniques |

---

### 📐 Mathematical Foundation

#### 🔹 1. Forward Propagation

For layer $l$, the pre-activation and post-activation values are:

$$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$$

$$a^{[l]} = g^{[l]}\!\left(z^{[l]}\right)$$

Where $W^{[l]}$ is the weight matrix and $g^{[l]}$ is the activation function of layer $l$.

#### 🔹 2. Activation Functions

| Function | Formula | Derivative |
|----------|---------|------------|
| **Sigmoid** | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ |
| **ReLU** | $\text{ReLU}(z) = \max(0, z)$ | $\mathbb{1}[z > 0]$ |
| **Tanh** | $\tanh(z) = \dfrac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ |
| **Leaky ReLU** | $\max(\alpha z,\ z),\ \alpha \approx 0.01$ | $\alpha$ if $z<0$ else $1$ |
| **Softmax** | $\text{softmax}(z_i) = \dfrac{e^{z_i}}{\sum_{j} e^{z_j}}$ | $\hat{y}_i(1-\hat{y}_i)$ |

#### 🔹 3. Loss Functions

**Mean Squared Error (Regression):**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2$$

**Binary Cross-Entropy (Binary Classification):**

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \Bigl[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \Bigr]$$

**Categorical Cross-Entropy (Multi-class):**

$$\mathcal{L}_{\text{CCE}} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

#### 🔹 4. Backpropagation — Chain Rule

The gradient of the loss w.r.t. weights at layer $l$:

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot \left(a^{[l-1]}\right)^T$$

$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} = \left(W^{[l+1]}\right)^T \delta^{[l+1]} \odot g'^{[l]}\!\left(z^{[l]}\right)$$

#### 🔹 5. Gradient Descent Update

$$W^{[l]} \leftarrow W^{[l]} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W^{[l]}}$$

$$b^{[l]} \leftarrow b^{[l]} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b^{[l]}}$$

#### 🔹 6. Batch Normalization

Normalize activations within a mini-batch:

$$\hat{z}^{(i)} = \frac{z^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad \text{then} \quad \tilde{z}^{(i)} = \gamma \hat{z}^{(i)} + \beta$$

Where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance, $\gamma$ and $\beta$ are learnable parameters.

#### 🔹 7. Dropout Regularization

During training, randomly zero out neurons with probability $p$:

$$a_{\text{dropped}} = a \odot \text{Bernoulli}(1-p) \cdot \frac{1}{1-p}$$

#### 🔹 8. Autoencoder Objective

The autoencoder learns a compact representation $z$ by minimizing reconstruction error:

$$\text{Encoder: } z = f_\phi(x), \quad \text{Decoder: } \hat{x} = g_\theta(z)$$

$$\mathcal{L}_{\text{AE}} = \|x - \hat{x}\|^2 = \|x - g_\theta(f_\phi(x))\|^2$$

#### 🔹 9. Min-Max Normalization

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

#### 🔹 10. Z-Score Standardization

$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

---

**🔑 Key Concepts:**
`Forward Propagation` · `Backpropagation` · `Activation Functions` · `Cross-Entropy Loss` · `Gradient Descent` · `Batch Normalization` · `Dropout` · `Autoencoders` · `Tensor Normalization`

---

## 🟢 CNN — Convolutional Neural Networks

<img src="https://img.shields.io/badge/Level-Intermediate-FFA726?style=flat-square"/> <img src="https://img.shields.io/badge/Type-Computer%20Vision-green?style=flat-square"/>

> *Spatial feature extractors that exploit local structure through shared weights and pooling.*

### 📁 Files

| File | Description |
|------|-------------|
| `CNN_Basic.ipynb` | Conv layers, pooling, flattening, and classification |
| `Image Classification/` | End-to-end image classification pipeline |
| `DATASET.zip` | Image dataset for training and testing |

---

### 📐 Mathematical Foundation

#### 🔹 1. 2D Discrete Convolution

A kernel (filter) $K$ slides over an input image $I$ to produce a feature map:

$$(I * K)[i,j] = \sum_{m=0}^{F-1} \sum_{n=0}^{F-1} I[i+m,\ j+n] \cdot K[m,n] + b$$

For a 3D input with $C$ channels:

$$(I * K)[i,j] = \sum_{c=1}^{C} \sum_{m=0}^{F-1} \sum_{n=0}^{F-1} I_c[i+m,\ j+n] \cdot K_c[m,n] + b$$

#### 🔹 2. Output Feature Map Size

Given input size $W$, filter size $F$, padding $P$, and stride $S$:

$$O = \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1$$

#### 🔹 3. Number of Parameters per Conv Layer

$$\text{Params} = (F \times F \times C_{\text{in}} + 1) \times C_{\text{out}}$$

where the $+1$ accounts for the bias term.

#### 🔹 4. ReLU Activation (applied after each Conv)

$$\text{ReLU}(z) = \max(0,\ z)$$

#### 🔹 5. Max Pooling

Reduces spatial dimensions by taking the maximum value in each pooling window $R_{i,j}$:

$$y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}$$

#### 🔹 6. Average Pooling

$$y_{i,j} = \frac{1}{|R_{i,j}|} \sum_{(m,n) \in R_{i,j}} x_{m,n}$$

#### 🔹 7. Receptive Field

The region of the original input that influences one neuron in layer $l$:

$$RF_l = RF_{l-1} + (F_l - 1) \cdot \prod_{k=1}^{l-1} S_k$$

#### 🔹 8. Flatten & Dense Classification Head

After convolutions and pooling, the spatial tensor is flattened:

$$\mathbf{v} = \text{Flatten}(A^{[L_{\text{conv}}]}) \in \mathbb{R}^{H' \times W' \times C'}$$

Then passed through fully connected layers ending in Softmax:

$$\hat{y} = \text{softmax}(W_{\text{fc}} \cdot \mathbf{v} + b_{\text{fc}})$$

#### 🔹 9. CNN Loss — Categorical Cross-Entropy

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

---

**🔑 Key Concepts:**
`2D Convolution` · `Feature Maps` · `Stride & Padding` · `Max / Avg Pooling` · `Receptive Field` · `Parameter Sharing` · `Flattening` · `Image Classification` · `Data Augmentation`

---

## 🔴 RNN — Recurrent Neural Networks

<img src="https://img.shields.io/badge/Level-Advanced-EF5350?style=flat-square"/> <img src="https://img.shields.io/badge/Type-Sequence%20Modeling%20%7C%20NLP-red?style=flat-square"/>

> *Networks with feedback connections — maintaining a hidden state that captures context across time steps.*

### 📁 Files

| File | Description |
|------|-------------|
| `Basic_RNN.ipynb` | Recurrent architecture, hidden states, and sequence processing |
| `Spam_Email_Dectechtion.ipynb` | 📧 Spam/Ham email detector with full NLP preprocessing |

---

### 📐 Mathematical Foundation

#### 🔹 1. RNN Hidden State (Recurrence Equation)

At each time step $t$, the hidden state $h_t$ is computed from the current input $x_t$ and the previous hidden state $h_{t-1}$:

$$h_t = \tanh\!\left(W_h \cdot h_{t-1} + W_x \cdot x_t + b_h\right)$$

In matrix form:

$$h_t = \tanh\!\left(\begin{bmatrix} W_h & W_x \end{bmatrix} \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_h\right)$$

#### 🔹 2. Output at Each Time Step

$$\hat{y}_t = \text{softmax}(W_y \cdot h_t + b_y)$$

#### 🔹 3. Total Loss — Sum Over All Time Steps

$$\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t(\hat{y}_t,\ y_t)$$

#### 🔹 4. Backpropagation Through Time (BPTT)

Gradient of loss w.r.t. weight $W_h$ accumulated over all time steps:

$$\frac{\partial \mathcal{L}}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W_h}$$

Using the chain rule through time:

$$\frac{\partial \mathcal{L}_t}{\partial h_k} = \frac{\partial \mathcal{L}_t}{\partial h_t} \cdot \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \frac{\partial \mathcal{L}_t}{\partial h_t} \cdot \prod_{i=k+1}^{t} W_h^T \cdot \text{diag}\!\left(1 - h_i^2\right)$$

#### 🔹 5. Vanishing Gradient Problem

As the sequence length grows, gradients through time can vanish:

$$\left\|\prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}\right\| \leq \left(\lambda_{\max} \cdot \gamma\right)^{t-k} \to 0 \quad \text{if } \lambda_{\max} \cdot \gamma < 1$$

Where $\lambda_{\max}$ is the largest eigenvalue of $W_h$ and $\gamma < 1$ is the activation derivative bound. This is why **LSTM** was invented.

#### 🔹 6. NLP Preprocessing Pipeline

**Tokenization:** Split text into tokens

$$\text{"spam email"} \rightarrow [\text{"spam"},\ \text{"email"}]$$

**Vocabulary Mapping:**

$$w \in V \rightarrow \text{index}(w) \in \{1, 2, \ldots, |V|\}$$

**TF-IDF (Term Frequency — Inverse Document Frequency):**

$$\text{TF-IDF}(t, d) = \underbrace{\frac{f_{t,d}}{\sum_{t'} f_{t',d}}}_{\text{TF}} \times \underbrace{\log\frac{N}{|\{d : t \in d\}|}}_{\text{IDF}}$$

---

**🔑 Key Concepts:**
`Recurrent Cells` · `Hidden State` · `BPTT` · `Vanishing Gradient` · `Sequence-to-Sequence` · `Tokenization` · `TF-IDF` · `Spam Detection`

---

## 🟣 LSTM — Long Short-Term Memory

<img src="https://img.shields.io/badge/Level-Advanced-EF5350?style=flat-square"/> <img src="https://img.shields.io/badge/Type-Gated%20Memory%20%7C%20Sentiment-purple?style=flat-square"/>

> *Gated memory cells that selectively remember and forget — solving the vanishing gradient problem of vanilla RNNs.*

### 📁 Files

| File | Description |
|------|-------------|
| `LSTM.ipynb` | 💬 Sentiment Analysis (Positive/Negative) using LSTM |
| `Fake/Real_news_Classification.ipynb` | 📧 Fake/Real news detector with full NLP preprocessing |

---

### 📐 Mathematical Foundation

At each time step $t$, the LSTM computes **four** key equations using the concatenated vector $[h_{t-1},\ x_t]$:

#### 🔹 1. Forget Gate — What to Erase from Memory

Decides what fraction of the previous cell state $C_{t-1}$ to forget:

$$f_t = \sigma\!\left(W_f \cdot [h_{t-1},\ x_t] + b_f\right)$$

$f_t \in (0, 1)^d$ — values close to $0$ mean "forget", close to $1$ mean "keep".

#### 🔹 2. Input Gate — What New Information to Store

Controls how much of the new candidate values to write into memory:

$$i_t = \sigma\!\left(W_i \cdot [h_{t-1},\ x_t] + b_i\right)$$

#### 🔹 3. Candidate Cell State — New Candidate Memory

$$\tilde{C}_t = \tanh\!\left(W_C \cdot [h_{t-1},\ x_t] + b_C\right)$$

#### 🔹 4. Cell State Update — Memory Update

Combines forgetting old memory and writing new memory:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Where $\odot$ denotes element-wise (Hadamard) multiplication.

#### 🔹 5. Output Gate — What to Output

Controls how much of the cell state to expose as the hidden state:

$$o_t = \sigma\!\left(W_o \cdot [h_{t-1},\ x_t] + b_o\right)$$

#### 🔹 6. Hidden State — Output of the LSTM Cell

$$h_t = o_t \odot \tanh(C_t)$$

---

#### 🔹 7. Summary: LSTM Gate Equations

$$\boxed{
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) & &\text{(Forget gate)}\\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) & &\text{(Input gate)}\\
\tilde{C}_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) & &\text{(Candidate)}\\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & &\text{(Cell state)}\\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) & &\text{(Output gate)}\\
h_t &= o_t \odot \tanh(C_t) & &\text{(Hidden state)}
\end{aligned}
}$$

#### 🔹 8. Total Trainable Parameters per LSTM Layer

$$\text{Params} = 4 \times \left[(d_x + d_h) \times d_h + d_h\right]$$

Where $d_x$ = input size, $d_h$ = hidden units. The factor of $4$ covers all four gates.

#### 🔹 9. Gradient Flow Through Cell State (No Vanishing!)

Because the cell state update is **additive**, gradients flow cleanly through time:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t \quad \Rightarrow \quad \frac{\partial \mathcal{L}}{\partial C_k} = \frac{\partial \mathcal{L}}{\partial C_t} \cdot \prod_{i=k+1}^{t} f_i$$

As long as $f_i \approx 1$ (forget gate open), gradients neither vanish nor explode.

#### 🔹 10. Word Embeddings (Input Representation)

Each word token $w$ is mapped to a dense vector via an embedding matrix $E \in \mathbb{R}^{|V| \times d}$:

$$x_t = E[w_t] \in \mathbb{R}^d$$

#### 🔹 11. Sentiment Classification Output

After processing the full sequence, the final hidden state $h_T$ is passed through a dense layer:

$$\hat{y} = \text{sigmoid}(W_{\text{out}} \cdot h_T + b_{\text{out}})$$

$$\text{Sentiment} = \begin{cases} \text{Positive} & \hat{y} \geq 0.5 \\ \text{Negative} & \hat{y} < 0.5 \end{cases}$$

---

**🔑 Key Concepts:**
`Forget Gate` · `Input Gate` · `Output Gate` · `Cell State` · `Hidden State` · `Hadamard Product` · `Gradient Highway` · `Word Embeddings` · `Sentiment Analysis`

---

## 🛠️ Tech Stack

<div align="center">

| 🏷️ Category | 🔧 Tools |
|-------------|---------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **NLP** | NLTK, Scikit-learn |
| **Math** | Linear Algebra, Calculus, Probability |
| **Environment** | Jupyter Notebook / Google Colab |

</div>

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Suman18-bit/Deep-Learning.git
cd Deep-Learning
```

### 2️⃣ Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn nltk jupyter
```

### 3️⃣ Launch Notebooks

```bash
jupyter notebook
```

> 💡 **Pro Tip:** Use **Google Colab** for free GPU acceleration — just upload the `.ipynb` files!

---

## 🗺️ Learning Path

```
🟢 BEGINNER
    │
    ▼
┌──────────────────────────────────┐
│  1. PERCEPTRON                   │
│  z = wᵀx + b  →  f(z) ∈ {0,1}  │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  2. ANN                          │
│  a[l] = g(W[l]·a[l-1] + b[l])  │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  3. CNN                          │
│  (I★K)[i,j] = ΣΣ I·K + b       │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  4. RNN                          │
│  hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b) │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  5. LSTM                         │
│  Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ        │
└────────────────┬─────────────────┘
                 │
                 ▼
🏁 DEEP LEARNING MASTERY ✨
```

| 🎯 Level | 📚 Topics | ⏱️ Time |
|---------|----------|---------|
| 🟢 Beginner | Perceptron, Binary Classification | 1–2 days |
| 🟡 Intermediate | ANN, Backprop, Autoencoders | 3–5 days |
| 🔵 Applied | CNN, Image Classification | 3–4 days |
| 🔴 Advanced | RNN, NLP, Text Classification | 3–4 days |
| 🟣 Expert | LSTM, Sentiment Analysis | 2–3 days |

---

<div align="center">

## 🤝 Contributing

Contributions and improvements are always welcome!
Check the [issues page](https://github.com/Suman18-bit/Deep-Learning/issues) or open a pull request.

---

## 📬 Connect

[![GitHub](https://img.shields.io/badge/GitHub-Suman18--bit-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Suman18-bit)

---

### ⭐ Found this useful? Please star the repo — it helps others discover it! ⭐

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>

*Made with ❤️, 📐 Math & ☕ by **Suman18-bit** — Happy Learning! 🚀*

</div>
