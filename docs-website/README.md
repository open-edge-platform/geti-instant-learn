# Test doucmentation website build

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator. Folders application/docs and library/docs
content is used.

## Installation

Make sure you have [node >= 18](https://nodejs.org/en) installed on your system. You can follow the instructions on the official node website
or use a package manager like homebrew.

## Build

```bash
just build-docs-web-site
```

This command generates static content into the `build` directory and serves as early preview before integrating documentation to web site.
