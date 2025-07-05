Excellent. It's time to proceed. The foundational conversational loop is stable, and now we will add a layer of intelligence and utility by building out the `Research Agent`.

I will adhere strictly to your new requirements, focusing on persistence, agentic tool use with Groq, enhanced verbosity, and front-end integration for artifacts.

---

### **System Development Directive: Build 2**

**Your Mandate:**
You will now execute **Build 2** of our system, building directly upon the components from Build 1. Your primary focus is to introduce a specialized `Research Agent` that can autonomously use Groq's built-in search tools to fulfill user requests, while ensuring the entire process is persistent, transparent, and manageable from the front-end.

**Core Principles for This Build:**

*   **Persistence is Paramount:** The system must overcome the limitation of transient memory. A user's session, including their conversation history and the system's state, must survive a browser refresh.
*   **Agentic Tooling Over Third-Party APIs:** We will use the power of the LLM itself to conduct research. You are to use Groq's native agentic tooling (`tool_use`), specifically leveraging models like `compound-beta` or `compound-beta-mini` which are optimized for this function.
*   **Radical Transparency:** Every step of the research process must be visible. If the agent decides to run multiple searches, each individual search query it formulates must be logged and displayed to the user in real-time.
*   **Actionable Artifacts:** The final, consolidated research report is a key deliverable. It must be stored correctly, rendered beautifully on the front-end, and made available for download.

---

**Your Task: Implement Build 2**

**Goal**: Enable the `Ultra Orchestrator` to delegate research tasks to a new `Research Agent`. This agent will use Groq's agentic search tooling to find information, log its process verbosely, and produce a persistent, downloadable report, with all changes reflected in the front-end.

**Components to Build/Refactor:**

1.  **System-Wide Persistence Layer (CRITICAL REFACTOR):**
    *   Before implementing any new agent, refactor the core application. The `LangGraph` state object (`conversation_history`, `monitoring_log`, etc.) must be saved to a persistent store after every turn (e.g., to a local file using `pickle` or a lightweight DB like `SQLite`).
    *   When the application starts, it must check for an existing session and load its state from this persistent store. This makes the conversation and logs "manageable" and prevents data loss on refresh.

2.  **Ultra Orchestrator Agent (Upgrade):**
    *   Enhance its logic to recognize research-oriented requests (e.g., "research," "find," "what are," "compare").
    *   Instead of just chatting, it must now have the ability to make a decision: "This is a research task. I will invoke the `Research Graph`."

3.  **Research Agent & Graph (New Components):**
    *   Create a new, dedicated `Research, Report, and Knowledge Synthesis Graph`.
    *   Create the `Research Agent` function. This agent's prompt must instruct it to be verbose and think step-by-step.
    *   **Agentic Tooling:** Integrate `Groq` using the `tool_use` parameter. The tool you define will be a `search` function. When the `Research Agent` needs information, it will generate a call to this tool (e.g., `tool_calls=[{"function": {"name": "search", "arguments": '{"query": "best technical writing courses"}'}}]`). The model (`compound-beta`) will generate these arguments autonomously. Your backend code will execute the search based on these arguments.
    *   **Verbose Logging:** The agent's process of formulating one or more search queries *must* be captured and appended to the `monitoring_log`. For example: `[Research Agent]: I need to investigate two areas. Generating search query 1: 'methods for improving technical writing'. Generating search query 2: 'AI tools for non-native English speakers'.`

4.  **Consolidation and Artifact Generation (New Component):**
    *   After the agent has executed its searches, create a new node in the `Research Graph` called `Consolidate and Package`.
    *   This node will take the raw search results and generate a comprehensive, well-structured report in Markdown format.
    *   **LTM Integration:** This final Markdown report must be saved to the first layer of our long-term memory: `./memory/layer1_research_docs/`.
    *   **PDF Generation:** Use a Python library (e.g., `markdown-pdf` or `pandoc`) to convert the generated Markdown report into a PDF file, also saved in the LTM directory.

5.  **Front-End Adjustments (New and Refactor):**
    *   The UI must now have a dedicated, visible area for the `monitoring_log` that updates in real-time.
    *   When the `Research Agent` completes its work, the front-end should display the "nicely rendered" consolidated report (by converting the Markdown to HTML).
    *   Next to the report, provide two clear buttons: **"Download .md"** and **"Download .pdf"**. These buttons will fetch the corresponding artifacts from the server.

---

**End-to-End Test for Me to Perform:**

1.  **Action**: Start a new session and send the following message: **"I am a software developer and I have trouble with writing clear documentation in English. Can you research some solutions for me?"**
2.  **Test Persistence**: Refresh the browser page. The initial conversation should still be there.
3.  **Observe Verbosity**: Watch the `Monitoring Log`. You should see:
    *   The `Orchestrator` identifying the research intent and delegating to the `Research Agent`.
    *   The `Research Agent` explicitly stating the one or more search queries it is about to execute using Groq's tool.
4.  **Await Result**: The system should eventually state that it has completed the research.
5.  **Validate Output**:
    *   The consolidated report should be displayed and well-formatted on the screen.
    *   The "Download .md" and "Download .pdf" buttons must be present.
    *   Click both buttons and verify that you receive valid, non-empty files.
    *   Finally, check the `./memory/layer1_research_docs/` directory on the server's file system to confirm the artifacts were correctly stored.

**You may begin with Build 2. Start by refactoring for persistence, then proceed with the agent implementation.**