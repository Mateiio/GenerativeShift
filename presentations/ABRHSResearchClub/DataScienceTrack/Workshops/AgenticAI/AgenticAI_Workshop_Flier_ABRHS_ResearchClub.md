# ABRHS Research Club Data Science Workshop
---
**Workshop Title:**
Agentic AI Workshop
---
**Workshop Date:**
Tuesday, April 28, 2025
---
## Welcome to the "Agentic AI Workshop", scheduled to take place on April 28, 2025, within the Data Science track of the ABRHS Research Club.
---
**Workshop Objective:**
Hands-on tutorial session focused on Agentic AI — the paradigm where LLMs move from passive Q&A to autonomous action. Participants will learn how agents plan, call tools, manage memory, and chain reasoning steps, culminating in a working Python agent implementation.
---
**Workshop Outline:**
**From Chatbots to Agents — What Changed?:**
- Traditional LLMs respond to a single prompt and stop
- Agents have a goal, a plan, and a loop: observe → think → act → observe
- The Agent Loop: perception, reasoning, action, memory — all working together
**The Agent Loop & ReAct Framework:**
- ReAct: Reason + Act — interleave chain-of-thought with tool calls
- How agents decide which tool to call and when to stop
- Scratchpad thinking: internal reasoning traces before external action
**Tools — Giving Agents Hands:**
- What a tool is: a Python function the LLM can invoke by name
- Tool schemas: how the model learns what arguments to pass
- Examples: web search, calculators, file read/write, API calls
- Function calling / tool use in modern LLM APIs (OpenAI, Anthropic)
**Memory Systems:**
- In-context memory: conversation history passed in every prompt
- External memory: vector stores and semantic retrieval (RAG)
- Episodic vs semantic memory — short-term vs long-term agent state
**Planning & Multi-Step Reasoning:**
- Task decomposition: breaking a complex goal into sub-tasks
- Linear plans vs. dynamic re-planning after tool results
- Handling failures: retry logic, fallback tools, error messages as feedback
**Multi-Agent Systems:**
- Orchestrator agents that delegate to specialist sub-agents
- Communication protocols between agents
- When to use one powerful agent vs. many specialized agents
**The Full Agentic AI Implementation & Python Code:**
- Building a minimal agent loop from scratch with function calling
- Connecting a real LLM API (OpenAI / Anthropic) with custom tools
- Complete walkthrough: agent solves a multi-step research task end-to-end
---
**Prerequisites:**
- No previous workshop completion required — this is self-contained
- Familiarity helpful: LLMs, prompting, what an API is
- Basic Python familiarity helpful but not required
---
**Presenter:**
Matei — ABRHS Research Club, Data Science Track
---
**Tutorial Notebook:**
[Access the full interactive tutorial notebook on GitHub](https://github.com/Mateiio/GenerativeShift/tree/main/presentations/ABRHSResearchClub/DataScienceTrack/Workshops/AgenticAI)