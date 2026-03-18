# ABRHS Research Club Data Science Workshop

---

**Workshop Title:**

Reinforcement Learning & Q-Learning Workshop

---

**Workshop Date:**

Tuesday, March 25, 2025

---

## Welcome to the "Reinforcement Learning & Q-Learning Workshop", scheduled to take place on March 25, 2025, within the Data Science track of the ABRHS Research Club.

---

**Workshop Objective:**

Hands-on tutorial session focused on fundamentals of Reinforcement Learning for data science projects, including the Q-Table, the Bellman Equation, and the full Q-Learning algorithm — implemented from scratch in Python using numpy and open-source Python libraries for scientific computing.

---

**Workshop Outline:**

**Concepts:**

- What is Machine Learning?
- The three branches of ML (Supervised, Unsupervised, Reinforcement)
- What is Reinforcement Learning?
- RL vocabulary: Agent, Environment, State, Action, Reward
- The RL loop
- Our environment: GridWorld (5×5 maze)

**The Q-Table:**

- What is a Q-Table and why does the agent need it?
- How Q-values are initialized and updated
- Reading the Q-Table: what each value means

**The Bellman Equation:**

- Intuition behind the update rule
- Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s′,a′) − Q(s,a) ]
- Learning rate (α) and discount factor (γ) explained
- Interactive demo: adjust parameters and see the effect live

**Explore vs Exploit (ε-greedy):**

- Why the agent must balance exploration and exploitation
- Epsilon-greedy strategy
- Epsilon decay: gradually shifting from explore to exploit

**The Full Algorithm & Python Implementation:**

- Step-by-step walkthrough of the complete Q-Learning loop
- Full Python code using numpy — written live during the workshop
- Watch the agent train in real time (300 episodes, live grid)
- Policy grid: visualizing what the trained agent has learned

---

**Prerequisites:**

- No prior machine learning knowledge required
- No advanced math background required
- Basic Python familiarity is helpful but not required

---

**Presenter:**

Matei — ABRHS Research Club, Data Science Track

*Full interactive tutorial notebook available on GitHub: Mateiio/GenerativeShift*
