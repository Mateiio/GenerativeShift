# ABRHS Research Club Data Science Workshop

---

**Workshop Title:**

Deep Q-Learning Workshop

---

**Workshop Date:**

Tuesday, March 25, 2025

---

## Welcome to the "Deep Q-Learning Workshop", scheduled to take place on March 25, 2025, within the Data Science track of the ABRHS Research Club.

---

**Workshop Objective:**

Hands on tutorial session focused on Deep Q-Learning (DQN), the extension of Q-Learning that uses neural networks to solve problems with large or continuous state spaces — including video games, robotics, and more.

---

**Workshop Outline:**

**Why Q-Learning Has Limits:**

- Q-Tables only work when the number of states is small and discrete
- Real environments like video games have astronomically large state spaces
- Solution: replace the Q-Table with a neural network that approximates Q-values

**Neural Networks — A Quick Introduction:**

- What a neural network is and how it transforms inputs into outputs
- Input layer, hidden layers, and output layer
- How the network learns through backpropagation and weight updates

**The DQN Architecture:**

- Online Network: the main network trained at every step
- Replay Buffer: stores past experiences for random batch sampling
- Target Network: a frozen copy used to compute stable training targets

**Experience Replay:**

- Why training on consecutive experiences causes instability
- How a circular replay buffer breaks temporal correlations
- Sampling random batches for more robust learning

**The Target Network:**

- Why the Bellman target becomes unstable if computed from the same network being trained
- How a periodically updated frozen network solves this
- target = R + gamma * max Q_target(s', a')

**The Full DQN Algorithm & Python Implementation:**

- Complete DQN loop walkthrough combining all components
- Full Python code using numpy and PyTorch
- Comparison of Q-Learning vs DQN side by side

---

**Prerequisites:**

- Completion of the Q-Learning Workshop recommended
- Familiarity with: Agent, Environment, State, Action, Reward, Q-Table, Bellman Equation
- Basic Python familiarity helpful but not required

---

**Presenter:**

Matei — ABRHS Research Club, Data Science Track

---

**Tutorial Notebook:**

[Access the full interactive tutorial notebook on GitHub](https://github.com/Mateiio/GenerativeShift/blob/main/presentations/ABRHSResearchClub/DataScienceTrack/Workshops/ReinforcementLearning_DeepQLearning/notebook/DeepQLearning_Workshop_ABRHS_ResearchClub.ipynb)
