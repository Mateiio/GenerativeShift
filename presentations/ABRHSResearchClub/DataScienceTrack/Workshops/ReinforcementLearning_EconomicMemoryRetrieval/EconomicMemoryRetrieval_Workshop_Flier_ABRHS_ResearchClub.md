# ABRHS Research Club Data Science Workshop

---

**Workshop Title:**

RL-Powered Economic Memory Retrieval Workshop

---

**Workshop Date:**

Tuesday, May 5, 2025 — ABRHS Research Club, Data Science Track

---

## Welcome to the "RL-Powered Economic Memory Retrieval Workshop", scheduled to take place on May 5, 2025 within the Data Science track of the ABRHS Research Club.

---

**Workshop Objective:**

Hands-on tutorial session focused on applying Reinforcement Learning to a real-world retrieval problem — teaching an agent to find the right economic fact from a noisy 490-entry memory bank, and understanding when and why RL outperforms simple similarity search.

---

**Workshop Outline:**

**Why Cosine Similarity Isn't Enough:**

- A 490-memory economics bank with overlapping vocabulary across inflation, GDP, interest rates, and crises
- Distractor memories score nearly as high as genuine facts
- Solution: train an RL agent to use richer signals than similarity alone

**The Economics Memory Bank:**

- 16 economic entities across 5 domains: macroeconomics, monetary policy, historical crises, economic theory, international trade
- Each fact stored in 5 paraphrased phrasings plus distractors and noise
- Why this structure makes the retrieval task genuinely hard

**Features — What the Agent Observes:**

- Cosine similarity, keyword overlap, entity name match, slot keyword match, rank decay
- 42-dimensional observation vector: 8 candidates × 5 features + 2 global features
- How richer features break ties that similarity cannot

**The Reward Function:**

- Shaped reward: gold match + entity + slot + sim + overlap − rank penalty
- Why partial credit accelerates learning vs binary reward
- Three interchangeable reward functions provided

**RL Algorithms Compared:**

- PPO (Proximal Policy Optimisation): on-policy neural network, stable convergence
- LinearBandit: ridge regression per action, fast and interpretable
- Random: lower-bound baseline showing what zero learning achieves
- Comparison against cosine, weighted, and top-rank non-RL baselines

**Full Python Implementation & Live Q&A:**

- Complete modular framework: swap embedding model, RL algorithm, or reward function independently
- Training a PPO agent in ~4 minutes on a standard laptop
- 25 real economics questions answered across all 5 domains

---

**Prerequisites:**

- Basic Python familiarity helpful but not required
- No prior economics knowledge assumed — all concepts explained from scratch
- Familiarity with RL basics (agent, environment, reward) is helpful but not required

---

**Presenter:**

Matei — ABRHS Research Club, Data Science Track

---

**Tutorial Notebook:**

[Access the full interactive tutorial notebook on GitHub](https://github.com/Mateiio/GenerativeShift)
