# ABRHS Research Club Data Science Workshop

**Workshop Title:** Advanced AI Agents Workshop

**Workshop Date:** [Day of the week, Month Day, Year]

Welcome to the "Advanced AI Agents Workshop", scheduled to take place on [Month Day, Year], within the Data Science track of the ABRHS Research Club.

**Workshop Objective:** The next level after Intro to AI Agents. Participants will run an agent that plans a task with several parts, asks for several tools at once and changes course when the data surprises it. Then they'll open up each upgrade, measure the agent's reliability with evals and error bars, and break it five new ways: lucky single runs, answers not backed by tools, caving to pushback, gaming the check, and orders hidden in retrieved documents. No API key or installs needed.

**Workshop Outline:**

**Run a Bigger Agent First:**

* Write a spring report from the same made-up school weather station, now March to May, plus its maintenance logs
* Watch the agent write a plan, request several tools in one step, and revise the plan when a reading of 69.8 °C makes no sense
* An answer that says what's known and what isn't

**Planning: Think Ahead, or One Step at a Time?:**

* Fixed workflow vs. one step at a time (ReAct) vs. plan-and-revise
* Missing-step errors, and what the research says about planning first
* Why the fixed workflow reported an impossible temperature

**The Upgraded Loop, Line by Line:**

* Parallel tool calls: several requests per reply, all results back in one message
* Timing it: the same answer in a third of the time
* Guards: rules your code enforces before any tool runs

**Tools, Next Level:**

* Designing tools for an agent: small results, problems made visible
* A sandboxed code tool, and why real agents run code in containers
* Hands-on: play the model with a parallel step, then write a code action

**Context Engineering:**

* Watching the context window grow over a 30-step task
* Trimming old steps, and how it made an agent confidently wrong
* Notes kept outside the context, and keyword search over station documents

**Where Advanced Agents Fail, and How to Catch It:**

* Evals: why one good run proves nothing, error bars on pass rates, and pass^k
* Grounding checks for answers not backed by tools; sycophancy under pushback
* An agent that edits the test instead of fixing the problem, and prompt injection through search, stopped by read-only and taint-tracking guards

**Teams, Next Level:**

* Subagents with their own context: smaller contexts, more total tokens
* Voting: when majority answers help, and when shared blind spots make them useless
* Try, check, retry loops, and why the checker must be out of the agent's reach

**Swap in a Real Model and Measure It:**

* The same loop running on Claude through the Anthropic API (optional, run by the presenter)
* Run the eval on the real model and read its interval
* Challenges to try at home

**Prerequisites:**

* Intro to AI Agents workshop recommended (or familiarity with the agent loop, tools and tool results)
* Comfortable reading basic Python: lists, dictionaries and functions
* Basic probability helpful: Section 6 works with pass rates and intervals
* Nothing to install and no API key needed: everything runs in Google Colab

**Presenter:** Matei — ABRHS Research Club, Data Science Track

**Tutorial Notebook:** [Access the full interactive tutorial notebook on GitHub](https://github.com/Mateiio/GenerativeShift/tree/main/presentations/ABRHSResearchClub/DataScienceTrack/Workshops/AdvancedAgents)
