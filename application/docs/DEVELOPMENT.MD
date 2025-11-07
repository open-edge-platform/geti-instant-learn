# Architecture Guidelines

## Overview

Our project follows a strict **3-layer architecture** designed for maintainability, testability, and clear separation of concerns. The entire system is governed by one fundamental principle:

> **The Rule:** Strict unidirectional dependency flow from top to bottom.
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

**Purpose:** Entry point for all external communication with the system.

**Responsibilities:**
- Input validation
- Response formatting
- HTTP/API protocol handling

**Components:**
- FastAPI endpoints
- Pydantic validation models
- Request/response handlers

**Dependency Rules:**
- ✅ **CAN** depend on: Runtime Layer, Configuration Layer
- ❌ **CANNOT** depend on: Nothing (top layer)

**Testing:** Unit tests with mocked Runtime and Configuration layers.

---

### 2. Runtime Layer

**Purpose:** Manages the application's live state and executes zero-shot learning pipelines.

**Responsibilities:**
- Pipeline lifecycle management
- Real-time state coordination
- Execution of ML workflows

**Components:**
- **PipelineManager**: Stateful singleton managing active pipeline(s). This is the only stateful singleton in the system. Ensure thread safety and document its lifecycle explicitly.
- **Control Services**: Services performing operations on the live system
  - Frame capture
  - Pipeline source state management
  - Runtime orchestration

**Dependency Rules:**
- ✅ **CAN** depend on: Configuration Layer
- ❌ **CANNOT** depend on: Presentation Layer

**Testing:** Unit tests with mocked Domain services.

---

### 3. Domain Layer

**Purpose:** Foundation of the application. Manages all data persistence, data-related business logic, and configuration.

**Responsibilities:**
- Data persistence and retrieval
- Business rule enforcement
- Configuration management
- Transaction coordination

**Components:**

#### Data Services
- Examples: `ProjectService`, `FrameConfigService`
- Orchestrate data retrieval and mutations
- Enforce business rules
- **Critical:** Data Services are the **only** components authorized to manage transaction boundaries. Never handle transactions in repositories, control services, or endpoints.

#### Repositories
- Lowest-level data access objects
- Examples: `IProjectRepository`, `IFrameRepository`
- Direct database/storage interaction
- **Always define as interfaces** (e.g., `IProjectRepository`) to enable easy mocking in tests, full isolation of Data Services, and flexibility in implementation swapping.

**Dependency Rules:**
- ✅ **CAN** depend on: Nothing (foundation layer)
- ❌ **CANNOT** depend on: Runtime Layer, Presentation Layer
- **Internal dependencies:**
  - Data Services → Repositories (via interfaces)
  - Repositories remain isolated from each other

**Testing:**
- Unit tests for services use mocked repositories.
- Integration tests for repositories use a real database.

---

## Dependency Flow Rules

### ✅ Allowed Dependencies
```
Presentation → Runtime
Presentation → Configuration
Runtime → Configuration
Data Services → Repositories
```

### ❌ Forbidden Dependencies
```
Configuration → Runtime
Configuration → Presentation
Runtime → Presentation
Repository → Repository
Repository → Data Services
```

**When in doubt:** Dependencies should always flow downward. If you need to communicate upward, consider using events, callbacks, or dependency injection.
