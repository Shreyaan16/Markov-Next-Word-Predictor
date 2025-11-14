# Markov Next Word Predictor & Analysis Dashboard

This project is a Python-based next-word predictor that uses a Markov chain model. It includes a [FastAPI](https://fastapi.tiangolo.com/) web application that serves as an interactive dashboard to computationally verify and visualize the core mathematical properties of Markov chains, as taught in a stochastic processes course.

## How the Next-Word Predictor Works

The predictor is built on the foundation of a first-order Markov chain, where the probability of the next word depends only on the current word.

1.  **State Space ($S$):** Each unique word in the training corpus is a "state" in our chain.
2.  **Transition Matrix ($P$):** The model builds a transition matrix $\mathbf{P}$, where $P_{ij}$ is the probability of transitioning from word $i$ to word $j$.
3.  **Building the Model:** The `build_model` function parses the text and calculates these probabilities empirically. It counts every time a word $j$ follows a word $i$ and normalizes this by the total number of transitions out of word $i$.
    $$P_{ij} = \frac{\text{Count}(i \to j)}{\sum_{k \in S} \text{Count}(i \to k)}$$
4.  **Prediction:** The `predict_next_word` function implements the **Markov Property**. It takes the current word (the `context`), looks up its corresponding row in the probability matrix, and returns the most likely next words (states).

## Analysis Dashboard

The core of this project is the dashboard, which provides computational proofs and visualizations of key mathematical concepts.

### 1. Chapman-Kolmogorov Equation
The dashboard provides a computational proof of the C-K equation. It verifies that the probability of an $n+m$-step transition is equal to the product of the $n$-step and $m$-step transition matrices. The implementation uses `np.linalg.matrix_power` to calculate $\mathbf{P}^{(n)}$ and $\mathbf{P}^{(m)}$ and then uses `np.dot` to show that $P_{ij}^{(n+m)} = (\mathbf{P}^{(n)} \mathbf{P}^{(m)})_{ij}$.

### 2. State Classification (Recurrent vs. Transient)
The model analyzes the state space graph to classify all states.
* **Communicating Classes:** It uses `networkx.strongly_connected_components` to find all communicating classes.
* **Classification:** It then checks if each class is "closed" by iterating through all nodes. If any node has a transition to a state *outside* its class, the entire class is marked as **transient**. Otherwise, it is marked as **recurrent**.

### 3. Periodicity
The code implements the mathematical definition of a state's period. The `get_period` function performs a graph search to find all possible path lengths $n$ that return to the starting state. It then computes the **Greatest Common Divisor (GCD)** of these lengths to find the period. This is used to verify the chain is aperiodic.

### 4. Stationary & Limiting Distributions
The dashboard demonstrates the long-run behavior of the chain in two ways:

**A. Stationary Distribution**
This analysis shows the stationary distribution $\pi$ calculated as the "long-run proportion". The `compute_stationary_distribution` function calculates this by finding the total count of each state $j$ and dividing by the total number of transitions in the entire corpus.

**B. Convergence to Limiting Distribution**
This analysis demonstrates the concept of a limiting distribution, just like the "weather example" from the lectures.
* The `analyze_convergence` function simulates a long **random walk** on the chain.
* The code plots the "running empirical probability" of the current state at each time step $n$.
* As $n$ (X-axis) increases, the probability (Y-axis) is shown to stabilize, or "converge," to a fixed value, visually proving that $\lim_{n \to \infty} P^{(n)}$ exists.

### 5. Random Walk (Text Generation)
The `generate_text` function is a practical simulation of a random walk. It starts at a state and, at each step, samples from the transition probability distribution `P(.|current_state)` to choose the next state, repeating the process to generate new text.