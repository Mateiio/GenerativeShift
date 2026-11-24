<div align="center">

<h1>Intro to AI Agents: Build the Simplest One</h1>

<p><em>A first look at AI agents: the simplest one there is, in about 20 lines of plain Python.<br>
No AI model, no API key and no installs required.</em></p>

<img src="assets/agent_loop.png" width="760" alt="The agent loop: the brain decides, the tool looks up a temperature, the notes remember it, and the loop repeats until the brain answers"/>

<p><a href="https://colab.research.google.com/github/Mateiio/GenerativeShift/blob/main/presentations/ABRHSResearchClub/DataScienceTrack/Workshops/IntroToAgents/notebook/IntroToAgents_Workshop_ABRHS_ResearchClub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="https://mateiio.github.io/GenerativeShift/presentations/ABRHSResearchClub/DataScienceTrack/Workshops/IntroToAgents/IntroToAgents_Workshop_ABRHS_ResearchClub.html"><img src="https://img.shields.io/badge/Open-interactive%20page-2563eb" alt="Open the interactive page"/></a></p>

</div>

---

## Contents

| # | Section | Topic |
|---|---------|-------|
| 1 | [What is an agent?](#1-what-is-an-agent) | The goal and the four parts of an agent |
| 2 | [The agent](#2-the-agent) | The whole agent in one cell |
| 3 | [Try it yourself](#3-try-it-yourself) | Change it and watch what it does |
| 4 | [From rules to AI agents](#4-from-rules-to-ai-agents) | Where a language model fits in |

> **Files in this folder**
>
> - `notebook/IntroToAgents_Workshop_ABRHS_ResearchClub.ipynb`: the notebook (runs in Colab, nothing to install)
> - `IntroToAgents_Workshop_ABRHS_ResearchClub.html`: a standalone version for reading or projecting ([view it online](https://mateiio.github.io/GenerativeShift/presentations/ABRHSResearchClub/DataScienceTrack/Workshops/IntroToAgents/IntroToAgents_Workshop_ABRHS_ResearchClub.html))
> - `IntroToAgents_README.md`: this file
> - `IntroToAgents_Workshop_Flier_ABRHS_ResearchClub.md`: the workshop flier
> - `assets/`: the agent loop diagram

---

## 1. What is an agent?

An agent works toward a goal in a loop. Ours has one goal: **find a city on our list that's at least 25 °C today.** It checks cities one at a time and decides after each one whether it's done. For that it needs four parts: a **tool**, a **brain**, **notes** and a **loop**.

## 2. The agent

```python
cities = ["Boston", "Seattle", "Miami", "Denver"]
temperatures = {"Boston": 4, "Seattle": 9, "Miami": 27, "Denver": -3}

def get_temperature(city):         # TOOL: looks something up
    return temperatures[city]

def brain(notes):                  # BRAIN: decides the next step
    for city in notes:
        if notes[city] >= 25:
            return "answer", city
    for city in cities:
        if city not in notes:
            return "check", city

notes = {}                         # NOTES: what the agent has found so far
while True:                        # LOOP: decide, act, remember, repeat
    action, city = brain(notes)
    if action == "answer":
        print("Go to", city)
        break
    notes[city] = get_temperature(city)
    print(city, "is", notes[city], "°C")
```

```text
Boston is 4 °C
Seattle is 9 °C
Miami is 27 °C
Go to Miami
```

It never looks up Denver: once the brain has its answer, the loop stops.

| # | Part | In the code |
|---|---|---|
| 1 | Tool | `get_temperature` looks up one city |
| 2 | Brain | Answers if a city in the notes is warm enough; otherwise picks a city it hasn't checked |
| 3 | Notes | `notes` remembers every temperature found so far |
| 4 | Loop | `while True` repeats decide, act, remember until the brain answers |

## 3. Try it yourself

- Put Miami first, and the agent checks one city.
- Change 25 to 0, and it sends you to Boston.
- Make Miami 20 and no city is warm enough, so the agent crashes. Students add a rule to the brain so it answers "nowhere" instead.

## 4. From rules to AI agents

In the AI agents behind ChatGPT and Claude, the brain is a language model: it reads the goal and the notes, and writes back the next step. The tool, the notes and the loop work the same way.

---

## Running the workshop

A suggested plan for about 25 minutes:

| # | Time | Section | Notes |
|---|---|---|---|
| 1 | 0:00 | 1. What is an agent? | Read the goal together; ask how a person would solve it |
| 2 | 0:05 | 2. The agent | Step through the run on the projector, then everyone runs the cell |
| 3 | 0:12 | 3. Try it yourself | The crash and its fix make a good discussion |
| 4 | 0:20 | 4. From rules to AI agents | Where a language model fits, then questions |

- On the projector, use the HTML page's **Next step** button to walk through the run one decision at a time.

---

## Further reading

- Anthropic (2024). [Building effective agents](https://www.anthropic.com/research/building-effective-agents). When to use an agent, and the patterns real ones are built from.
- Yao et al. (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). *ICLR.* The decide, act, observe loop with a language model as the brain.

---
*Tutorial produced as part of the ABRHS Research Club workshop series. The temperatures are invented for teaching.*
