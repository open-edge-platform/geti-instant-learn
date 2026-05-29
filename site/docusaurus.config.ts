import type * as Preset from '@docusaurus/preset-classic';
import type { Config } from '@docusaurus/types';
import { themes as prismThemes } from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

// GITHUB_REPOSITORY env var is set by GitHub Actions
const GITHUB_REPOSITORY = process.env.GITHUB_REPOSITORY || 'open-edge-platform/geti-instant-learn';
const [organizationName, projectName] = GITHUB_REPOSITORY.split('/');

const config: Config = {
  title: 'Geti Instant Learn',
  favicon: 'img/favicon.svg',

  // Set the production url of your site here
  url: `https://${organizationName}.github.io`,
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: `/${projectName}/`,

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName, // Usually your GitHub org/user name.
  projectName, // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: false,
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      disableSwitch: false,
      defaultMode: 'light',
    },
    navbar: {
      title: 'Geti Instant Learn',
      logo: {
        alt: 'Geti Logo',
        src: 'img/geti-logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Library Docs',
        },
        {
          type: 'docSidebar',
          docsPluginId: 'applicationDocs',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Application Docs',
        },
        {
          href: `https://github.com/${organizationName}/${projectName}`,
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Library',
              to: '/docs/library/introduction',
            },
            {
              label: 'Application',
              to: '/docs/application/introduction',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: `https://github.com/${organizationName}/${projectName}`,
            },
          ],
        },
        {
          title: 'Legal',
          items: [
            {
              label: 'License',
              href: `https://github.com/${organizationName}/${projectName}/blob/main/LICENSE`,
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Open Edge Platform. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        path: '../library/docs',
        routeBasePath: 'docs/library',
        sidebarPath: './sidebars.library.ts',
        editUrl: `https://github.com/${organizationName}/${projectName}/tree/main/library/docs`,
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'applicationDocs',
        path: '../application/docs',
        routeBasePath: 'docs/application',
        sidebarPath: './sidebars.application.ts',
        editUrl: `https://github.com/${organizationName}/${projectName}/tree/main/application`,
      },
    ],
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        highlightSearchTermsOnTargetPage: true,
        searchBarShortcutHint: false,
      },
    ],
  ],
};

export default config;
