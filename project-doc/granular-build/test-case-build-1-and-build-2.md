Of course. This is the most critical phase: validating the implementation against the design. A robust testing plan will ensure that each build not only works in isolation but also integrates seamlessly with the previous one.

Here is a comprehensive, end-to-end testing script you can execute from your front-end. It is designed to be run sequentially, verifying the capabilities of Build 2 first, and then using that successful state to verify the new capabilities of Build 3.

---

### **Prerequisites for Testing**

Before you begin, please ensure the following conditions are met:

1.  **Front-End is Running:** Your front-end application is active and can communicate with the back-end.
2.  **Back-End is Running:** The back-end server with the code for both Build 2 and Build 3 is running.
3.  **Real API Keys Configured:** Your back-end environment has a valid **Groq API key** loaded. The tests are designed to make real calls, so this is essential.
4.  **Memory Directory Exists:** The `./memory/` directory, containing the subdirectories `./memory/layer1_research_docs/` and `./memory/layer2_build_artifacts/`, exists and is writable by the back-end process.

---

### **Test Plan 1: Verifying Build 2 - The Research & Persistence Core**

**Objective:** To confirm that the system can handle a research request, use Groq's agentic tooling, maintain state across a refresh, produce verbose logs, and deliver persistent, downloadable artifacts.

**Step-by-Step Instructions:**

| Step | Action (What you do on the front-end) | Expected Outcome & Observations (What you see on the front-end) |
| :--- | :--- | :--- |
| **1. Session Persistence Test** | 1. Open the web application. <br> 2. Type a simple greeting: **"Hello, system."** <br> 3. Wait for the system to respond. <br> 4. **Refresh the browser page (F5/Cmd+R).** | The conversation, including your "Hello, system" message and the AI's reply, must still be visible. This confirms the persistence layer is working. |
| **2. Trigger Research Agent** | In the same session, enter the specific prompt: **"I am a software developer and I have trouble with writing clear documentation in English. Can you research some solutions for me?"** | The system should acknowledge the request. The conversation will likely pause as it begins processing. |
| **3. Monitor Verbose Logs** | Watch the "Monitoring Log" panel on your UI closely. | You should see a real-time stream of messages appear, detailing the system's internal state. Look for specific key events like: <br> - `[Orchestrator]: User request identified as a research task.` <br> - `[Orchestrator]: Invoking Research, Report, and Knowledge Synthesis Graph.` <br> - `[Research Agent]: Formulating search queries using Groq tool_use.` <br> - `[Research Agent]: Executing search with query: '...'` (This proves the agentic tooling is being called). <br> - `[Research Agent]: Consolidating results into a final report.` |
| **4. Verify Artifact Delivery** | Wait for the system to complete the task. It should post a message in the chat indicating the research is done. | 1. The main chat/content area should now display a **well-formatted summary** of the research findings (the report rendered from Markdown). <br> 2. Two clickable buttons must be visible: **"Download .md"** and **"Download .pdf"**. |
| **5. Test Artifact Functionality** | 1. Click the **"Download .md"** button. <br> 2. Click the **"Download .pdf"** button. | 1. Your browser should download a valid Markdown file. Open it and verify it contains the full research report. <br> 2. Your browser should download a valid PDF file. Open it and verify it is a properly rendered version of the report. |
| **6. Back-end File Check** | *(With your developer hat on)* Check the back-end file system. | Navigate to the `./memory/layer1_research_docs/` directory. It must contain the `.md` and `.pdf` files you just downloaded. |

**If all steps above are successful, Build 2 is confirmed to be working correctly. Do not proceed until you have passed this stage.**

---
