 "walk-me-through" mode is developed here C:\Users\Admin\Documents\GitHub\sentient-core\sentient-core-wmt-prototype

## Documents of the project
1: the stage "walk-me-through" description: 
2: the "walk-me-through" technical implementation:
3: reference and documents of stacks uses (not in any order and I may miss some you should reasonably research and recommend if needed):
a) autogen: https://github.com/microsoft/autogen 
b) Groq API and their highly related documents (remember to research of AI models for which is capable of what and you must use some of the llma models) https://console.groq.com/docs/overview ; https://console.groq.com/docs/agentic-tooling ; https://console.groq.com/docs/api-reference#chat-create ; https://github.com/groq/groq-api-cookbook
c) uses of google gemini models (make use of OpenAI compatibility for not having to install additional google gemini SDK but you have to install OpenAI SDK) - the google models are specifically used for image, video and audio generation so they are used for certain steps and for particular agents https://ai.google.dev/gemini-api/docs ; https://ai.google.dev/gemini-api/docs/openai ; 
d) other documents from the framework (lang-graph, langchain, pydantic): https://langchain-ai.github.io/langgraph/concepts/why-langgraph/ ; https://python.langchain.com/docs/introduction/ ; https://pypi.org/project/pydantic/
f ) E2B and their implementation for code interpreter, front end render, desktop uses, making artifacts to download and viewable on front-end: https://e2b.dev/docs ; https://github.com/e2b-dev/e2b ; https://github.com/e2b-dev/code-interpreter ; https://github.com/e2b-dev/desktop ; https://e2b.dev/docs/quickstart/connect-llms ; https://github.com/e2b-dev/e2b-cookbook (cookbook with exmples)
https://e2b.dev/docs/code-interpreting/analyze-data-with-ai ; https://e2b.dev/docs/code-interpreting/analyze-data-with-ai/pre-installed-libraries ; https://e2b.dev/docs/code-interpreting/create-charts-visualizations ; 
https://e2b.dev/docs/code-interpreting/create-charts-visualizations/interactive-charts ; https://github.com/e2b-dev/ai-analyst/ (a sample git to data analyst) ; https://e2b.dev/docs/code-interpreting/streaming (support streaming for output)
d) fetch.ai - our stacks' sponsor for hackathon see if you can include some of their technology into our project: https://fetch.ai/docs


## Sample and model repos from GitHub Repos

1) Fragment: the one like Anthropic Artifacts or Google Canvas which AI agent generate user's request into 4 stack frameworks Nextjs, Python for data analysis, Streamlit and Gladio
2) Cookbook, E2B SDK core, code interpreter and desktop uses are their SDK open-source and cookbook which include many examples: 
3) Archon - a repo that demonstrate a framework that agents that build agents (you can see and if there are nay implementation for us)

## Our first step-build 

Goal: to completely build an multi-agent RAG flow of walk me through laying as foundation for further improvement, more complex and nuances build 

### Brief description from the user workflow perspectives

1. user start with natural language varied from "greeting" to very vague an inaccurate like "build me a website to sell phones", "I want an app that teach me English", "the database of my inventory is here", "make a research onto technology impacts to human brain", to something doubtful and not knowing "here is my sales data what can you do", to some with more details, or even very specific like build a certain app using what stacks and doing in which manner and so on. And on top of that they may also upload assets like images, document, pdf file; and these are either provided with context or without context. What's more? The platform is also promised to provide a more tailored and robust solution to users from various industries, working in different positions of cross-departments in workplace environment, handling every aspects of their requirement through sophisticated and robust multi-agentic RAG system of multiple memory and high-level techniques of langgraph, langchain, and nowadays agentic tooling and using tools so that digital solutions are non-barrier with technological literate and satisfaction to arrays of users no matter what their tech-expertise levels are. 

> [!NOTE] Please provide feasible solution so that 
> This first step is highly reasoning with agents in charge to either: 
> - trying to understand with enhancement of prompt, redirect users with politeness for off-topic prompt, tailored and encouraging responses with guidance and instructions that are on-points and act as helpers for a more specific and best-practice of inputs - These must take into consideration of knowing what tasks the users really want to accomplish, their fields of work, position, departments, and so on)
> - or making a chain of thoughts and conduct a short agentic search by reasoning from user input, understanding from context and keywords -> this may be done by self-reasoning and decision to initiate a quick chain of searching for relevant knowledge footprints (using tools) -> to response users more cohesively. Using online research tools to browse internet resources to compare, contrast and analytically give response from natural languages (meaning users don't actually include links or address exact technical words and terms - AI agent needs to automatically know) with reasoning to outline what users' actually want regarding to their task complexity, industry of work, department of work, actual tasks completion requirement (not just app, but may be research, brainstorming data analysis or any sub-tasks like UI generation, wireframe generation, formal document preparation, slides generation etc)
> -  Or right away initiate the following chains that involve with level below sub-graph (langgraph - which you may need to look at Langchain and LangGrapho, and Groq API to make these more specific and nuances to cover what is called a one-shot prompt from user). This third situation derives from rather specific and simple build which Orchestrator agent can detect (by reasoning ) and confirm by monitor agent that it is sufficient to execute a chain of actions carried by sub-graph and pipelines. One of the example form such prompt is : "Can you build me simple chatbot using prebuilt AI chat UI (like the ones used by ChatGPT of OpenAI, Perplexity and Gemini. I want the chat platform is integrated with Google Gemini model gemini-2.5-flash with 'image understanding/ vision capabilities and audio understanding' and 'generate audio' capabilities enabled'. As for the chatbot behaviors, functions and features I want it acts like an IELTS English tutor for intermediate Vietnamese learners to practice IELTS speaking. user will input the questions or context of practice and the chatbot will response as a tutor and IELTS examiner to guide user through different questions with feedbacks and scoring following the official" user will use voice to record and AI tutor will response in text and playable with audio".
> 	- For detail prompt as above (of course it is just one example you must develop the orchestrator and the monitor agents to intelligently dissect and take regard action because like I said above there are multiple and nuances of industries, tasks, expertise of tech, complexity of tasks and department ) the following chain actions will get initiated: NOTICE: be aware of states management, event-driven, some can take turns some can be simultaneous, remember to really understand about the frameworks uses, the AI models, the system instruction prompt the key stacks for the following flow are from Langgraph, Groq API, LangChain, E2B (of various SDK) so please research and reason at high level
> 	- research sub-graph of multiple agents will have to look for and collaboratively communicate with each other so that it can first:
> 		- Gather latest (2025) relevant parts of documents of required stacks
> 		- synthesize into dot md documents of consolidated knowledge -> store at shared database (with robust chunking and relational, indexed, vectorized technique so that the latter graph of agents groups can carry on)
> 	- These database knowledge gets pass to architecting piplelines where multiple agents will dissect build into plans drafts of the most major plans like PRD, high-level plan, task-breakdown etc 
> 	- After this  the results of these synthesize document will get summed up to output to front-end to the user which is downloadable as PDF or Markdown and ask for confirmation of the stack choices and foundational build plans
> 	- if all plans agree with then pass to next stage, if not the above graphs and pipelines may get reinitiate for modification mode based on users' feedbacks
> 	- design, UX, UI graph/pipeline group will retrieve synthesized knowledge + historical chat prompt of so-far conversation of user to provide these
> 		- the workflow depicts in wireframe and mock UI - these are built with simple react Nextjs, HTML and CSS, Streamlit and Gladio for python (making these as artifacts that's viewable and interactable using E2B and their code-interpreter, and templates uses and streaming, you can learn from [e2b-dev/fragments: Open-source Next.js template for building apps that are fully generated by AI. By E2B.](https://github.com/e2b-dev/fragments)
> 		- Continue to modify base on user feedback - user feedback at this stage may also involve planning and architecting pipelines to fit with the design UX UI workflow
> 		- if everything is agree on the context and knowledge (in this stage get distributed to long-term database which must be classified, chunked, indexed, making relational data, and vectorized) to then planning and architecting graph/pipeline groups of agents get to work with synthesized knowledge to produce action build plan which show details in phases, tasks and sub-tasks and code samples
> 		- The builder will start building the app

### In-case it is not a one shot prompt

- more instruction and guidance responses steps and monitoring agents and orchestrator should be in-charge to direct and guide user to provide more information regarding to the aforementioned matrix
- But it should be steps like mentioned above but with more guidance and direction

### Key notices

- a very robust and sophisticated design of multi-agent agentic RAG that make use of these 4 keys stacks
- **LangGraph and LangChain**: you may need to look at their Streaming, Persistence, Durable execution, Memory of short-term and long-term, Context, Tools, Human-in-the-loop, Sub-graphs, Evaluation, Multi-agents, Data management (please look into these very deeply, for Langchain Tools and Toolkits, integration with Pydantic and Groq API, Vector Stores, Embedding, Retrievers...)
- **Groq API**: use latest model from llama 3.3, and new 4.0, pay attention to their agentic toolings, streaming capabilities, structured output - some of their compound and compound-mini though beta can be used for some of very useful tools uses that we don't need to use a third party
- **Tool Uses:** must be very robust and making best uses of agentic and retrieval system - for example how AI agent make a chain of search based on previous synthesized knowledge with high reasoning or use combination of tools to produce much accurate and sophisticated results and not just mere single search - look at the core values of the platforms  "sophisticated and robust multi-agentic RAG system of multiple memory and high-level techniques of langgraph, langchain, and nowadays agentic tooling and using tools so that digital solutions are non-barrier with technological literate and satisfaction to arrays of users no matter what their tech-expertise levels, departments and industries of works, or their tasks"
- **E2B**: with SDK like computer uses, SDK core, SDK code interpreter, E2B cookbook examples, E2B fragments, E2B code streaming, E2B templates will give you a more comprehensive guides for how to stream agents' work to front-end, how to present artifacts (or called as fragments in E2B) as sample mini apps (written in React NextJs, Streamlit, Gladio, Python etc), or make code snippets and downloadable documents, images, and codes
- **Multi models and switching profiles/prompts:** the techniques of multiple models switching to handle various cases (like using google gemini for Image and Video generation but mainly Groq API will be in use)
- Be very cautious of structured and JSON output as they need to be consistent and properly rendered to front-end with proper libraries and dependencies or stacks - or else it will breaks the app
- Extra libraries needed and you must very concern about these libraries as never provide the version without searching for the latest on online with suffices like "2025" because your cut-off knowledge maybe outdated
- Most documents needed for the stacks and guides are indexed in this IDE environment so please search for it
- the real platform use Supabase and their extensions however for this prototype build please find the replacement that can quickly establish for both short-term, long-term memory
- Harness the power of all the stacks to make the below capable - make sure all the natural language of all languages must be understood and response in input languages (except for coding) to the users regardless if the turn is included of tools uses or not
- Implementation of monitor and regulator agents to make sure the purposes of the platform is used correctly and not being exploited by:
	- to have the agent resolve user's needs in a more comprehensive input including what to build, for what purposes working in which department and what industry etc... meaning the overview that is sufficient for what's about to happen next. If the conversation get to 5th turn but going nowhere the agent will warn politely and the IP of user will get banned from continue the conversation from the 7th turn and they can only refresh to start over. Or if the conversation is off-tracking or off-topic it must be redirected with warning
4) The logic here is the Orchestrator must help user by guided following up response that contextually throughout the conversation by add on details and intelligently use search tools if possible to iterate a complete boarding overview for what's coming next




## The results I would like to have for this unit development
demonstration of the foundational walk-me-through step of the platform
1) A clear structure of directories that shows graphs, sub-graph and pipelines, tools, services, state, main agent,  tests 
2) A sufficient front end that on the main side (portion 6 mentioned below) can rendered the following (in form of artifacts like Anthropic one and inline of the Agent natural language chat with their toggle or raw output). By making use of E2B the following "artifacts" can be output and rendered correctly as their intentions:
	1) code files
	2) complete functional mini-app (for UI, for simple apps) can be rendered as preview (establish by using sandbox environment and template builds for each type of coding ) - these are as mentioned above
		1) simple React NextJs app (often include front-end pages with simple CRUD API, and local simple database with implementation of AI-powered chatbot), Streamlit and Gladio for Python build, and Python with popular pre-installed libraries for data analysis purposes -> these are downloadable into zip files with instruction how to run on development channels and use with users' local environment (use files storing and editing capabilities of E2B sandbox for this)
		2) Python code execution for data-oriented and chart-oriented tasks that rendered into table, and interactive charts/graphs/diagrams
		3) Images for wireframe
		4) Images for assets generation
		5) documents exportable in either PDF or markdown
3) a streamlit  dashboard of chat like of portion 4:6 
	1) The 4 portion: showing input API key, selection of models and branches (these will be switch automatically depends of the first step chat input but if I want to quickly switch and see how the variation works it should show too). Below it is the dialog windows show action log of turn summarizing what services, tools or actions that the orchestrator used and which states  it is. This also shows errors for debug
	2) The 6 portion: the chat flow that requires orchestrator agent responses in natural language (and he must first response in natural language before using tools) + a toggle to show raw tools/services/actions output and of course the rendered output if agent use tools like research or computer use to then allow download to various form (markdown, PDF, docx, or png image)