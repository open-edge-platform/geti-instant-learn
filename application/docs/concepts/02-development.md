# Architecture Guidelines

## Overview

The application follows a strict **3-layer architecture** with unidirectional dependency flow.

```
┌─────────────────────────────────┐
│    API (Presentation) Layer     │  ← Entry point
├─────────────────────────────────┤
│    Runtime Layer                │  ← Live state & execution
├─────────────────────────────────┤
│    Domain Layer                 │  ← Foundation (data & config)
└─────────────────────────────────┘
```

---

## Layer Definitions

### 1. API (Presentation) Layer
**Purpose:** Entry point for external communication.
- **Responsibilities:** Input validation, response formatting, HTTP handling.
- **Components:** FastAPI endpoints, Pydantic models.
- **Dependencies:** Can import `Runtime` and `Domain`.

### 2. Runtime Layer
**Purpose:** Manages live state and ML pipeline execution.
- **Responsibilities:** Pipeline lifecycle, real-time coordination.
- **Components:**
  - `PipelineManager`: Singleton for active pipeline state.
  - Control Services: Frame capture, runtime orchestration.
- **Dependencies:** Can import `Domain`. Cannot import `API`.

### 3. Domain Layer
**Purpose:** Foundation for data persistence and business logic.
- **Responsibilities:** Data retrieval, business rules, transactions.
- **Components:**
  - **Data Services**: Orchestrate logic and manage transactions (e.g., `ProjectService`).
  - **Repositories**: Interfaces for data access (e.g., `IProjectRepository`).
- **Dependencies:** No external dependencies.

---

## Key Rules

1.  **Unidirectional Flow:** Dependencies **must** flow top-to-bottom (API → Runtime → Domain).
2.  **Transactions:** Only **Data Services** in the Domain Layer manage transaction boundaries.
3.  **Interfaces:** Always define Repositories as interfaces to enable mocking.
4.  **State:** `PipelineManager` is the **only** allowed stateful singleton.

## Testing Strategy

-   **Unit Tests:** Mock dependencies (e.g., API tests mock Runtime; Runtime tests mock Domain).
-   **Integration Tests:** Test Repositories with a real database.
