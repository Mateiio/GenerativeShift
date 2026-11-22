# ABRHS Research Club Data Science Workshop

**Workshop Title:** Intro to AI Agents Workshop

**Workshop Date:** [Day of the week, Month Day, Year]

Welcome to the "Intro to AI Agents Workshop", scheduled to take place on [Month Day, Year], within the Data Science track of the ABRHS Research Club.

**Workshop Objective:** Hands-on introduction to AI agents: programs where a language model works in a loop, choosing tools and acting on the results until a task is done. Participants will run a working agent in the first ten minutes, open up its loop line by line, play the model's part themselves, and then break the agent five different ways to learn where agents fail and how to guard against it. No API key or installs needed.

**Workshop Outline:**

**Run an Agent First:**

* Run a working agent on weather data from a made-up school station, data no AI model has ever seen
* Watch every step of the loop: thought → tool request → observation → answer
* Why the agent needs tools: the answer isn't in any model's training data

**What Makes Something an Agent?:**

* One model call vs. a fixed workflow vs. an agent: who decides the steps?
* When not to use an agent, and when one is worth the cost
* Why this "agent" isn't the reinforcement-learning kind from the Q-Learning workshop

**The Agent Loop, Line by Line (ReAct):**

* ReAct: Reason + Act, where the model plans a step, asks for a tool, reads the result and repeats
* The model only writes text; your code runs the tools
* Stop reasons and step limits: how the loop knows when to stop

**Tools — Giving Agents Hands:**

* A tool is a Python function plus a schema the model reads
* Tool requests and tool results in Claude's message format
* Hands-on: play the model yourself, then build a new tool

**Memory:**

* The messages list is the agent's working memory, re-sent on every step
* Where memory lives (context window vs. files and databases) and what it holds (the task, facts, past events)
* A notes file that survives between conversations, and retrieval (RAG) in brief

**Where Agents Fail, and How to Guard Against It:**

* Stuck loops and step limits; tool errors as feedback
* The "negative rainfall" trap: a confident answer from misread data
* Why small error rates add up over long tasks
* Prompt injection, and defenses: fewer tools and human approval

**Teams of Agents:**

* Orchestrator, pipeline and reviewer patterns
* What the research says: review helps most when it checks something real
* When one agent beats many

**Swap in a Real Model:**

* The same loop running on Claude through the Anthropic API (optional, run by the presenter)
* Compare the real model's steps with the scripted run
* Challenges to try at home

**Prerequisites:**

* No previous workshop completion required — this is self-contained
* Familiarity helpful: what an LLM is, prompting, what an API is
* Basic Python familiarity helpful but not required
* Nothing to install and no API key needed: everything runs in Google Colab

**Presenter:** Matei — ABRHS Research Club, Data Science Track

**Tutorial Notebook:** [Access the full interactive tutorial notebook on GitHub](https://github.com/Mateiio/GenerativeShift/tree/main/presentations/ABRHSResearchClub/DataScienceTrack/Workshops/IntroToAgents)
