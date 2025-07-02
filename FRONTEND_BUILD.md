# Sentient-Core: Next.js Front-End Build

This document tracks the tasks required to build the Next.js 15, React 19, and Shadcn/UI front-end for the Sentient-Core application.

## Completed Tasks

- [ ]

## In Progress Tasks

- [ ] **Phase 1: Project Setup & Initialization**
  - [ ] Task 1.1: Create a new directory for the front-end application (`sentient-core-fe`).
  - [ ] Task 1.2: Initialize a Next.js 15 project with TypeScript, Tailwind CSS, and App Router.
  - [ ] Task 1.3: Manually configure Shadcn/UI in the project.
  - [ ] Task 1.4: Establish a scalable directory structure (`components`, `lib`, `app`, etc.).
- [ ] **Phase 2: Backend-Frontend Integration**
  - [ ] Task 2.1: Configure FastAPI backend to accept requests from the Next.js front-end (CORS).
  - [ ] Task 2.2: Set up environment variables in Next.js for the backend API URL.
- [ ] **Phase 3: Core UI Implementation - Chat Interface**
  - [ ] Task 3.1: Build the main page layout.
  - [ ] Task 3.2: Create the core chat components using Shadcn/UI (MessageList, Message, ChatInput).
  - [ ] Task 3.3: Implement client-side state management for the chat interface.
- [ ] **Phase 4: API Client & E2E Workflow**
  - [ ] Task 4.1: Develop an API client service to communicate with the FastAPI `/invoke` endpoint.
  - [ ] Task 4.2: Integrate the API client into the chat input component.
  - [ ] Task 4.3: Ensure the end-to-end flow is functional (send prompt/image, receive and display response).

## Future Tasks

- [ ] Implement user authentication.
- [ ] Build out the "Documents Drawer" and memory layer interface.
- [ ] Develop the "Pro Mode" IDE view.

## Implementation Plan

### Phase 1: Project Setup

I will start by scaffolding a new Next.js project in a separate `sentient-core-fe` directory. I will then manually add the necessary configuration for Shadcn/UI, as its interactive CLI is not suitable for this environment. This includes setting up `tailwind.config.ts`, `postcss.config.js`, and `components.json`.

### Phase 2: Integration

Next, I will update the FastAPI `main.py` to include `CORSMiddleware`, allowing the front-end, which will run on a different port, to make API calls to the backend. I will also add a `.env.local` file to the Next.js project to manage the backend URL securely.

### Phase 3 & 4: UI and Workflow

With the setup complete, I will build the user-facing chat interface using pre-built components from Shadcn/UI where possible to ensure speed and quality. This will involve creating a main page, structuring the layout, and building the components for displaying messages and handling user input. Finally, I will write the client-side `fetch` logic to connect to our backend, completing the primary user interaction loop.

## Relevant Files

- `sentient-core-fe/` - Root directory for the new Next.js front-end.
- `sentient-core/app/main.py` - To be modified for CORS.
- `FRONTEND_BUILD.md` - This tracking file.
