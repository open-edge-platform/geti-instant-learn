# Geti Instant Learn Documentation Site

This is a Docusaurus-based documentation site for the Geti Instant Learn project.

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
npm install
# or
yarn install
```

### Local Development

```bash
npm start
# or
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```bash
npm run build
# or
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

```bash
npm run deploy
# or
yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

## Project Structure

- `../library/docs/` - Source for library documentation
- `../application/docs/` - Source for application documentation
- `src/` - Source files including CSS and custom components
- `static/` - Static assets (images, etc.)

This site is configured to read docs directly from the original folders above.

## Documentation

- [Docusaurus](https://docusaurus.io/) - Documentation framework
- [Markdown Guide](https://docusaurus.io/docs/markdown-features) - Markdown support in Docusaurus
