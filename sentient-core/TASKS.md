# Walk-Me-Through Feature Implementation

This document outlines the tasks required to build the foundational "Walk-Me-Through" feature. This feature will guide users through a sophisticated, multi-agent workflow to define and prototype applications, using Streamlit for the front-end and a backend powered by LangGraph, Groq, E2B, and Fetch.ai.

## 🔴 Phase 0: Research & Synthesis (Completed)

- [x] Analyze `blue-print.md` and `walk-me-through-build.md` to establish core requirements.
- [x] Research `fetch.ai` and `Agentverse` to define a decentralized agent communication strategy.
- [x] Research LangGraph, E2B, and Groq for latest best practices and integration patterns.

## 🟡 Phase 1: Project Setup & Core Services

- [ ] Task 1.1: Create the complete project directory structure.
- [ ] Task 1.2: Initialize `pyproject.toml` with all necessary dependencies (streamlit, uagents, langgraph, groq, e2b, etc.).
- [ ] Task 1.3: Create a `.env` file from `.env.example` and implement `core/config.py` to securely load API keys.
- [ ] Task 1.4: Implement the basic Streamlit application layout in `app/main.py` with the 4:6 column structure (sidebar for controls, main area for chat).

## 🟡 Phase 2: Foundational Agent & Orchestration Graph

- [ ] Task 2.1: Define the `Orchestrator` agent in `core/agents/orchestrator_agent.py` using the `uagents` library.
- [ ] Task 2.2: Implement the main `Orchestration` graph in `graphs/orchestration_graph.py` using LangGraph. This graph will define the agent's internal thought process.
- [ ] Task 2.3: The initial node in the graph will be "Process User Input," which analyzes the user's prompt to determine the required next step (e.g., clarify, research, design).
- [ ] Task 2.4: Connect the Streamlit chat input to the Orchestration graph. User messages will trigger the graph, and the initial text response from the agent should be streamed back to the UI.

## 🟡 Phase 3: Research Sub-System (Agent + Sub-Graph)

- [ ] Task 3.1: Implement web search tools in `core/tools/` using the provided Tavily and Exa API keys.
- [ ] Task 3.2: Define the `Research` agent in `core/agents/research_agent.py` as a `uAgent`.
- [ ] Task 3.3: Create a `research_subgraph.py` in `graphs/` that defines the research process: `take_topic` -> `search_web` -> `synthesize_findings`.
- [ ] Task 3.4: Integrate the `Research` agent into the main orchestration. The `Orchestrator` will delegate research tasks by sending a message to the `Research` agent's address. The results will be sent back and displayed in the Streamlit UI as a markdown artifact.

## 🟡 Phase 4: E2B Integration & Code Artifact Generation

- [ ] Task 4.1: Implement a robust `E2BService` in `core/services/e2b_service.py` to manage sandboxes.
- [ ] Task 4.2: Create a `Code` agent (`core/agents/code_agent.py`) responsible for generating and executing code.
- [ ] Task 4.3: Add a tool to the `Code` agent that allows it to write a file (e.g., `app.py`) to an E2B sandbox.
- [ ] Task 4.4: Implement functionality to retrieve the generated file from the sandbox and present it as a downloadable artifact in the UI.

## ⚪️ Phase 5: Advanced Artifacts (UI & Design)

- [ ] Task 5.1: Create a `Design` agent (`core/agents/design_agent.py`).
- [ ] Task 5.2: Implement a tool for the `Design` agent to generate wireframes using an image generation model (e.g., Google Gemini Vision).
- [ ] Task 5.3: Display the generated image artifact in the chat flow.
- [ ] Task 5.4: (Advanced) Implement a tool using E2B Fragments to generate a live preview of a simple HTML/CSS page within the E2B sandbox and embed it in the Streamlit UI.

## ⚪️ Phase 6: Testing, Deployment & Refinement

- [ ] Task 6.1: Develop unit tests for all services and tools.
- [ ] Task 6.2: Create integration tests for the primary agent workflows (e.g., a full research loop).
- [ ] Task 6.3: Write a `README.md` with detailed setup and run instructions.

## Implementation Plan

This section will be filled out as development progresses, detailing the architecture and data flow decisions.

### Relevant Files

This list will be populated as files are created and modified.

- `sentient-core/`
    - `app/`
    - `core/`
    - `graphs/`
    - `tests/`
    - `.env`
    - `pyproject.toml`
    - `TASKS.md` 