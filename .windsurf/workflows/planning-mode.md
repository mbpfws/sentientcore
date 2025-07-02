---
description: Build the complete action plan for development
---

Phase 1: Task Comprehension & Contextualization

Upon receiving a request for a feature build plan, the AI agent must first establish a complete understanding of the task and its surrounding context. This phase ensures foundational clarity before any action is taken.

1.  **Analyze Feature Scope and Technical Landscape:**
    *   **1.1. Technology Stack Identification:** Determine the primary technology stack(s) involved (e.g., backend, frontend, API, mobile, database) and the specific programming languages, frameworks, and runtime environments required.
    *   **1.2. Project Type Assessment:** Ascertain if the feature is an enhancement to an existing codebase/structure (requiring integration and adherence to established patterns) or a greenfield development (allowing for new architectural decisions).
    *   **1.3. Prior Context & Dependencies:** Identify any dependencies on previous work, existing modules, or historical project context that might influence the current feature build.
    *   **1.4. Documentation Availability:** Verify the provision of relevant documentation, including internal codebase documentation, user-provided specifications, or external references for specific stacks, frameworks, libraries, SDKs, and dependencies.

2.  **Prompt & Context Validation:**
    *   **2.1. Clarity and Completeness Check:** Rigorously assess the user's input prompt and any provided context for clarity, completeness, and technical accuracy.
    *   **2.2. Insufficient Context Handling:** If the input prompt or provided context is ambiguous or insufficient, the agent *must* initiate a context-gathering process:
        *   **2.2.1. Internal Knowledge Retrieval:** Utilize available MCP server tools to retrieve relevant information:
            *   `ByteRover`: For accessing past chat sessions and conversational context.
            *   `Desktop-commander-mcp`: To list directories, read relevant code files, and parse `.md` documentation within the provided codebase.
            *   `Knowledge-Graph MCP`: To query for relationally stored knowledge pertinent to the project or domain.
        *   **2.2.2. User Elaboration Request:** If internal knowledge retrieval proves insufficient, the agent *must* explicitly and precisely request further elaboration or additional context from the user, detailing the specific information needed.

### Phase 2: Research, Knowledge Acquisition & Synthesis

Once the task is understood, the agent proceeds to gather and synthesize all necessary information, prioritizing best practices and proposing optimizations.

1.  **External Research for Best Practices:**
    *   **2.1. Continuous Learning & Optimization:** The agent *must never assume* and always leverage internet and web browsing tools to conduct comprehensive research. This ensures the proposed implementation is based on the most up-to-date, efficient, and best-in-class approaches, leading to optimal choices for stacks, frameworks, and libraries.
    *   **2.2. Proactive Optimization Proposals:** If, based on evidential context and research, the agent detects inaccuracies, inefficiencies, or sub-optimal approaches in the user's prompt or existing patterns, it *must* immediately propose superior alternatives. Each proposal requires clear explanation, robust reasoning, and comparative analysis.

2.  **Systematic Knowledge Synthesis:**
    *   **2.3. Purpose of Synthesis:** Once sufficient evidence, context, documentation, and code-level knowledge are gathered, the agent will perform knowledge synthesis. The resulting synthesized documents serve as the foundational material for the upcoming build plan.
    *   **2.4. Document-Based Knowledge Processing:**
        *   **2.4.1. Retrieval & Incremental Synthesis:** Retrieve and systematically arrange documents related to required stacks, libraries, dependencies, and SDKs (from inbuilt context, provided URLs, or `.md` files). These documents are to be read in manageable chunks. As each chunk is processed, the agent will generate synthesized knowledge documents. Due to context window limitations, this process must be iterative, focusing solely on information relevant to the upcoming build plan.
        *   **2.4.2. Structured Output:** All synthesized documents *must* be generated in various `.md` file formats and named using a systematic and hierarchical convention to facilitate efficient future retrieval and referencing.
    *   **2.5. Codebase-Based Knowledge Processing:**
        *   **2.5.1. Deep Code Analysis:** If provided with sample codebases (from GitHub or local workspaces), the agent *must* employ the `Desktop-commander` MCP tool. This tool is essential for listing and reading relevant code files to the deepest levels of the provided code structure, enabling a thorough understanding of essential features, patterns, and existing implementations.
        *   **2.5.2. Code-Derived Synthesis:** Knowledge derived from code analysis must be synthesized in the same systematic and hierarchical manner as document-based knowledge, ensuring it is readily consumable for plan generation.
    *   **2.6. Consolidated Knowledge Base:** After generating structured knowledge from both documentation and codebases, the agent will collectively synthesize these individual documents. This final consolidation step involves cherry-picking and integrating only the most pertinent information into a unified, systematic, and hierarchical knowledge base directly applicable to the feature build plan.

### Phase 3: Comprehensive Action Plan Generation

With a robust understanding and synthesized knowledge base, the agent proceeds to generate a detailed, actionable, and test-driven build plan.

1.  **Hierarchical Plan Structure:**
    *   **3.1. Phased Breakdown:** The action plan *must* be broken down into clear, distinct phases, each comprising hierarchical and manageable tasks and sub-tasks.
    *   **3.2. Structured Code Architecture:** The plan *must* include a firm and structured code tree. This involves clearly stating the proposed names for each directory, sub-directory, and individual code file. Strict consistency in naming conventions is paramount.

2.  **Detailed Implementation Guidance:**
    *   **3.3. Build Explanations:** For each task and sub-task, provide a detailed explanation of the build process. This includes accurate and up-to-date pseudo-code or model code snippets, accompanied by comprehensive explanations of correlational and connected files or code parts.

3.  **Integrated Testing Strategy:**
    *   **3.4. Continuous Unit Testing:** The action plan *must* inherently incorporate a high frequency of unit testing. Progress to subsequent tasks or phases is contingent upon the successful completion and passing of all preceding unit tests.
    *   **3.5. Plan Validation & Iteration:** The agent is required to validate the efficacy of its generated plan by actually building sample code snippets or modules within designated test directories (co-located with the plan documentation). The agent *must* reflect on the outcomes of these sample builds and modify the action plan accordingly if any discrepancies or improvements are identified.