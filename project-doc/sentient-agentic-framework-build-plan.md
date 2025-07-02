# Sentient Agentic Framework: A Comprehensive Build Plan

This document outlines the phased development plan for building a highly autonomous, multi-agent RAG system as envisioned. The goal is to create a sophisticated, stateful, and scalable framework that can handle complex user requests from initial vague ideas to fully realized, documented, and testable solutions.

---

## Guiding Principles

- **Autonomy & Intelligence:** Agents should make independent, reasoned decisions. The system will be driven by natural language understanding, not rigid commands.
- **Modularity & Specialization:** Each agent and graph will have a clearly defined, specialized role.
- **Transparency & Monitoring:** The system's operations will be transparent to the user through a dedicated Monitoring Agent and a verbose UI.
- **Stateful & Resilient:** The entire workflow will be stateful, allowing for interruptions, user feedback, and iterative refinement.
- **User-Centric Collaboration:** The system will act as a collaborative partner, guiding the user from ambiguity to clarity.

---

## Phased Development Plan

### Phase 1: Foundational Architecture & Models (The Bedrock)

This phase focuses on establishing the core data structures and architectural patterns that will support the entire system.

-   **Task 1.1: Consolidate & Enhance Pydantic Models.**
    -   **Action:** Review `core/models.py` and `core/enhanced_models.py`.
    -   **Goal:** Merge the best of both into a single, cohesive set of models in `core/models.py` that will serve as the single source of truth for the entire system's state. This includes robust definitions for `SystemState`, `AgentState`, `EnhancedTask`, `MemoryEntry`, and `SessionContext`. `enhanced_models.py` will be deprecated and removed.

-   **Task 1.2: Create Core Agent Skeletons.**
    -   **Action:** Create the Python files and basic class structures for all new and enhanced agents.
    -   **Files:**
        -   `core/agents/ultra_orchestrator.py`
        -   `core/agents/monitoring_agent.py`
        -   `core/agents/architect_planner_agent.py`
        -   `core/agents/frontend_developer_agent.py`
        -   `core/agents/backend_developer_agent.py`
        -   `core/agents/coding_agent.py`
    -   **Goal:** Establish the architectural footprint of our agent team.

-   **Task 1.3: Define the Main Workflow Graph.**
    -   **Action:** Create a new graph file: `graphs/sentient_workflow_graph.py`.
    -   **Goal:** This will be the top-level LangGraph graph that orchestrates the entire multi-agent workflow, starting with the `UltraOrchestrator` and `MonitoringAgent`.

### Phase 2: The `UltraOrchestrator` & Monitoring Loop (The Brain & Nervous System)

This phase brings the core decision-making and feedback loop to life.

-   **Task 2.1: Implement the `UltraOrchestrator`'s Core Logic.**
    -   **Action:** Develop the primary intelligence of the system within the `UltraOrchestrator` class.
    -   **Capabilities:** Natural Language Understanding (NLU) for intent detection, conversation management (clarification, redirection), and task breakdown logic. It will be designed to handle multilingual input and image context.

-   **Task 2.2: Implement the `MonitoringAgent`.**
    -   **Action:** Develop the agent responsible for observing state changes across the graph.
    -   **Goal:** To provide the verbose, step-by-step updates to the front-end, making the system's internal workings transparent to the user.

-   **Task 2.3: Implement the Core Orchestration Loop.**
    -   **Action:** In `sentient_workflow_graph.py`, wire the initial nodes together.
    -   **Flow:** `User Input` -> `UltraOrchestrator` (makes a decision) -> `MonitoringAgent` (reports the decision) -> `[Route to next step]`.

### Phase 3: Research Sub-System (The Knowledge Gatherer)

This phase builds out the first specialized capability: advanced research.

-   **Task 3.1: Enhance the `ResearchAgent`.**
    -   **Action:** Refine the existing `ResearchAgent` to officially support the three distinct modes: `Knowledge`, `Deep`, and `Best-in-Class`.
    -   **Goal:** Create a powerful, adaptable research tool.

-   **Task 3.2: Create the Research Sub-Graph.**
    -   **Action:** Create `graphs/research_subgraph.py` to manage the multi-step research processes.
    -   **Goal:** Encapsulate the complexity of research tasks, from planning to execution to synthesis.

-   **Task 3.3: Integrate the Research Sub-Graph.**
    -   **Action:** The `UltraOrchestrator` will delegate to this sub-graph when a research task is initiated.

### Phase 4: The 4-Layer Memory System (The Long-Term Memory)

This phase implements the persistent knowledge base.

-   **Task 4.1: Design and Implement the `MemoryService`.**
    -   **Action:** Create `core/services/memory_service.py`.
    -   **Goal:** This service will manage the 4-layer memory system (Knowledge Synthesis, Build Conversation, Codebase Knowledge, Stack Dependencies). It will handle storage, retrieval, chunking, and vectorization.

-   **Task 4.2: Integrate Memory Across All Agents.**
    -   **Action:** All agents will use the `MemoryService` to read from and write to the long-term memory.

### Phase 5: Architecture & Build Sub-Systems (The Creators)

This phase develops the agents responsible for designing and building the software.

-   **Task 5.1: Implement `ArchitectPlannerAgent` & Sub-Graph.**
    -   **Action:** Develop the agent that synthesizes research and conversations into PRDs, tech specs, and high-level plans. Create its corresponding sub-graph.
-   **Task 5.2: Implement Developer Agents & Build Graph.**
    -   **Action:** Implement the `FrontendDeveloperAgent`, `BackendDeveloperAgent`, and `CodingAgent`. Create the `build_subgraph.py` where they will collaborate.

### Phase 6: UI & E2B Integration (The Interface)

This phase connects the powerful backend to an intuitive user interface.

-   **Task 6.1: Overhaul the Streamlit UI.**
    -   **Action:** Redesign `app/main.py` to visualize the new agentic workflow.
    -   **Features:** A pane for the `MonitoringAgent`'s logs, dynamic display of generated artifacts (documents, prototypes), and interactive feedback mechanisms.
-   **Task 6.2: Integrate E2B for Sandboxing and Previews.**
    -   **Action:** Use E2B to securely run code generated by the `CodingAgent`.
    -   **Goal:** Render live previews of prototypes (e.g., Next.js, Streamlit apps) directly within the user interface.

---

I will now proceed with **Phase 1, Task 1.1**: Consolidating the Pydantic models to create a solid foundation for this new architecture. 