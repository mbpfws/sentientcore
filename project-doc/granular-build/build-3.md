

---

### **System Development Directive: Build 3**

**Your Mandate:**
You will now execute **Build 3**, which builds directly upon the tested and running components from Builds 1 and 2. Your primary mission is to introduce the `Architect Planner Agent` and its corresponding graph. This will transition the system's capability from pure information gathering to intelligent solution design, creating a tangible development plan from the research artifacts.

**Core Principles for This Build:**

*   **Proactive Conversational Guidance:** The system must not automatically proceed from research to planning. It must demonstrate intelligence by proactively asking the user for permission to take the next logical step, thus keeping the user in control.
*   **Structured, Actionable Planning:** The output of this stage is not just more information; it's a structured, actionable plan—a Product Requirements Document (PRD)—that will serve as the blueprint for the subsequent development phase.
*   **Tiered Memory Expansion:** We will now activate and populate the second layer of our long-term memory, demonstrating the system's ability to store different kinds of knowledge in a structured, hierarchical manner.

---

**Your Task: Implement Build 3**

**Goal**: After the `Research Agent` successfully generates a report, the `Ultra Orchestrator` will ask the user for permission to create a plan. Upon approval, it will delegate the task to a new `Architect Planner Agent`, which will synthesize the research into a PRD and save it as a new, persistent artifact in the appropriate memory layer.

**Components to Build/Refactor:**

1.  **Ultra Orchestrator Agent & Graph (Upgrade):**
    *   **Stateful Decision Making:** Enhance the main `Orchestrator Graph` state with a new flag, e.g., `research_complete: boolean`. This flag will be set to `true` after the `Research Agent` finishes.
    *   **Proactive Guidance Logic:** Add a new conditional edge to the graph. After research is complete, the `Ultra Orchestrator`'s next response *must* be to ask the user for permission to proceed. For example: `"The research is complete and the report is available. Based on these findings, shall I proceed with creating a high-level plan and Product Requirements Document for a solution?"`
    *   **Invocation Logic:** If the user agrees, the orchestrator must then invoke the new `Project Architecture Graph`.

2.  **Architect Planner Agent (New Component):**
    *   Create a new agent function for the `Architect Planner`. Its prompt will instruct it to act as a senior product architect.
    *   Its primary input will be the consolidated research report artifact generated in Build 2. It must be able to load this file from `./memory/layer1_research_docs/`.
    *   Its primary output will be a comprehensive, well-structured PRD in Markdown format. The PRD should outline key features, user stories, and high-level technical considerations.

3.  **Project Architecture Graph (New Component):**
    *   Create a new, dedicated `LangGraph` instance for this workflow.
    *   Its flow will be straightforward for now: `Input: Research Report` -> `Node: Call Architect Planner to Generate PRD` -> `Node: Save PRD to LTM and Package for User`.

4.  **Long-Term Memory (LTM) - Layer 2 (New Component):**
    *   Activate the second tier of the memory system. The generated PRD (both the `.md` and `.pdf` versions) must be saved to a new, distinct directory: `./memory/layer2_build_artifacts/`. This is separate from the research reports.

5.  **Front-End Adjustments (Refactor):**
    *   The conversational UI must correctly display the Orchestrator's new, proactive question.
    *   Upon creation of the PRD, the front-end must render the new artifact just as it did for the research report, displaying the formatted Markdown and providing **"Download .md"** and **"Download .pdf"** buttons for the PRD.

---

**End-to-End Test for Me to Perform:**

1.  **Action (Phase 1 - Research):** Start a new session and trigger the flow from Build 2. User: **"I am a software developer struggling with writing clear documentation. Please research solutions for me."**
2.  **Checkpoint (Verify Build 2 Works):** Wait for the system to complete the research and present the downloadable research report.
3.  **Action (Phase 2 - The New Step):** The system must now ask a question like, **"Shall I create a Product Requirements Document for a solution based on this research?"** You should respond: **"Yes, please proceed."**
4.  **Observe Verbosity**: Watch the `Monitoring Log`. It should clearly show the `Ultra Orchestrator` invoking the `Project Architecture Graph` and the `Architect Planner Agent` being activated.
5.  **Validate Output (Verify Build 3):**
    *   The system should announce that the PRD is complete.
    *   The new PRD artifact should be rendered on the screen.
    *   The download buttons for the PRD must work for both formats.
    *   Finally, check the server's file system. You must see the original research report in `./memory/layer1_research_docs/` AND the new PRD in `./memory/layer2_build_artifacts/`. This confirms the tiered memory is functioning correctly.

**You may now begin with Build 3. Focus first on upgrading the Orchestrator's logic to enable this new, crucial conversational step.**