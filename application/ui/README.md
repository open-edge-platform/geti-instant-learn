# Geti Prompt UI

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)
[![Node Version](https://img.shields.io/badge/node-24.2.0-brightgreen)](https://nodejs.org/)
[![NPM Version](https://img.shields.io/badge/npm-11.3.0-blue)](https://www.npmjs.com/)

A modular UI for Geti Prompt, a framework for few-shot visual segmentation using visual prompting techniques. This UI enables experimentation with different algorithms, backbones, and project components for finding and segmenting objects from just a few examples.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Available Scripts](#available-scripts)
- [Development](#development)
- [Testing](#testing)
- [Linting & Formatting](#linting--formatting)
- [Building](#building)
- [Contributing](#contributing)
- [License](#license)

---

## Prerequisites
- [Node.js v24.2.0](https://nodejs.org/)
- [npm v11.3.0](https://www.npmjs.com/)

## Installation

Clone the repository and install dependencies:

```bash
npm install
```

> **Note:** On the first installation, UI packages from [open-edge-platform/geti](https://github.com/open-edge-platform/geti) will be cloned automatically.

---

## Available Scripts

All commands are run from the `ui` directory unless otherwise specified.

| Script                       | Description                                             |
|------------------------------|---------------------------------------------------------|
| `npm start`                  | Start the UI development server (http://localhost:3000) |
| `npm run server`             | Start the backend server (http://localhost:9100)        |
| `npm run dev`                | Start the UI and backend servers                        |
| `npm run build`              | Build the app for production                            |
| `npm run preview`            | Preview the production build locally                    |
| `npm run lint`               | Run ESLint                                              |
| `npm run lint:fix`           | Auto-fix lint issues                                    |
| `npm run format`             | Format all files with Prettier                          |
| `npm run format:check`       | Check formatting with Prettier                          |
| `npm run type-check`         | Run TypeScript type checking                            |
| `npm run test:unit`          | Run unit tests with Vitest                              |
| `npm run test:unit:ui`       | Run unit tests in interactive UI mode                   |
| `npm run test:unit:coverage` | Run unit tests with coverage report                     |
| `npm run test:unit:watch`    | Run unit tests in watch mode                            |
| `npm run test:component`     | Run component tests with Playwright                     |
| `npm run build:api`          | Download and generate OpenAPI types                     |
| `npm run cyclic-deps-check`  | Check for circular dependencies                         |

---

## Development

To start the development, you need to have a running UI dev server and backend server. You can run them using:

```bash
npm run dev
```

or

To start UI dev server:
```bash
npm start
```

To start a backend server:
```bash
npm run server
```

The app will be available at [http://localhost:3000](http://localhost:3000).

---

## Testing

- **Unit tests:**
  ```bash
  npm run test:unit
  ```
- **Unit tests (UI mode):**
  ```bash
  npm run test:unit:ui
  ```
- **Unit tests (watch):**
  ```bash
  npm run test:unit:watch
  ```
- **Unit tests (coverage):**
  ```bash
  npm run test:unit:coverage
  ```
- **Component tests (Playwright):**
  ```bash
  npm run test:component
  ```

---

## Linting & Formatting

- **Lint:**
  ```bash
  npm run lint
  ```
- **Lint (auto-fix):**
  ```bash
  npm run lint:fix
  ```
- **Format:**
  ```bash
  npm run format
  ```
- **Format check:**
  ```bash
  npm run format:check
  ```

---

## Building

- **Production build:**
  ```bash
  npm run build
  ```
- **Preview production build:**
  ```bash
  npm run preview
  ```

---

## Type Checking

- **Type check:**
  ```bash
  npm run type-check
  ```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the [Apache 2.0 License](../LICENSE).

---