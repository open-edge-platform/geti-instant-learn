import type * as Preset from '@docusaurus/preset-classic';
import type { Config } from '@docusaurus/types';
import { themes as prismThemes } from 'prism-react-renderer';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

import { GETI_INSTALLER_LOCATION } from './config';
import { InstallGetiScript } from './src/plugins/install-script';
import { LLMsTxt } from './src/plugins/llms-text';
import ScalarDocusaurus from './src/plugins/scalar';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)
const DESCRIPTION = `As part of Intel's Open Edge Platform, Geti™ enables anyone from domain experts to data scientists to rapidly develop production-ready AI models.

**Build AI From End-To-End**
From data collection up to model deployment: Geti™ makes it easy for any user to go from data to model in minimum amount of time.

**Models Optimized For Intel Hardware**
Train and deploy your computer vision model with optimized performance for Intel hardware. Deploy your model in the precision that meets your needs.

**Free & Open Access**
Geti™ is part of Intel's Open Edge Platform, an open source ecosystem that provides scalable edge solutions. Use Geti™ the way you want: deploy the software for free on your own hardware with our installer, or access the source code on GitHub.`;

const SIDEBARS_CONFIG = {
};

const config: Config = {
    title: 'Geti™',
    tagline: 'Powerful AI for everyone',
    favicon: 'img/geti-favicon-32.webp',

    // Set the production url of your site here
    url: process.env.SITE_URL ?? 'http://localhost:3000',
    // Set the /<baseUrl>/ pathname under which your site is served
    // For GitHub pages deployment, it is often '/<projectName>/'
    baseUrl: process.env.BASE_URL ?? '/',

    // GitHub pages deployment config.
    // If you aren't using GitHub pages, you don't need these.
    organizationName: 'intel-innersource', // Usually your GitHub org/user name.
    projectName: 'https://github.com/intel-innersource/applications.ai.geti.geti-website/tree/main/', // Usually your repo name.

    onBrokenLinks: 'throw',
    onBrokenMarkdownLinks: 'throw',

    markdown: {
        mermaid: true,
    },

    // Even if you don't use internationalization, you can use this field to set
    // useful metadata like html lang. For example, if your site is Chinese, you
    // may want to replace "en" with "zh-Hans".
    i18n: {
        defaultLocale: 'en',
        locales: ['en'],
    },

    future: {
        experimental_faster: true,
        v4: true,
    },

    presets: [
        [
            'classic',
            {
                docs: {
                    sidebarPath: './sidebars.ts',
                    remarkPlugins: [remarkMath],
                    rehypePlugins: [rehypeKatex],
                },
                blog: false,
                theme: {
                    customCss: './src/css/custom.css',
                },
            } satisfies Preset.Options,
        ],
    ],

    themeConfig: {
        image: 'img/get-logo.svg',
        colorMode: {
            disableSwitch: false,
            respectPrefersColorScheme: true,
        },
        navbar: {
            title: '',
            logo: {
                alt: 'Geti™ logo',
                src: 'img/geti-logo.svg',
            },
            items: [
                {
                    type: 'dropdown',
                    sidebarId: 'applications',
                    label: 'Applications',
                    position: 'left',
                    items: [
                        {
                            label: 'Geti Instant Learn',
                            to: '/docs/instant-learn/get-started',
                        },
                    ],
                },
                {
                    href: 'https://github.com/open-edge-platform/geti',
                    label: 'GitHub',
                    position: 'right',
                },
            ],
        },
        tableOfContents: {
            maxHeadingLevel: 4,
        },
        footer: {
            style: 'light',
            links: [
                {
                    title: 'Legal',
                    items: [
                        {
                            label: 'Terms of Use',
                            href: 'https://docs.openvino.ai/2024/about-openvino/additional-resources/terms-of-use.html',
                        },
                        {
                            label: 'Responsible AI',
                            href: 'https://www.intel.com/content/www/us/en/artificial-intelligence/responsible-ai.html',
                        },
                    ],
                },
                {
                    title: 'Privacy',
                    items: [
                        {
                            label: 'Cookies',
                            href: 'https://www.intel.com/content/www/us/en/privacy/intel-cookie-notice.html',
                        },
                        {
                            label: 'Privacy',
                            href: 'https://www.intel.com/content/www/us/en/privacy/intel-privacy-notice.html',
                        },
                    ],
                },
            ],
            copyright: `Copyright © ${new Date().getFullYear()} Intel Corporation
                Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.
                Other names and brands may be claimed as the property of others.`,
        },
        prism: {
            additionalLanguages: ['bash'],
            theme: prismThemes.github,
            darkTheme: prismThemes.dracula,
        },
        zoom: {
            selector: '.zoom img',
            background: {
                light: 'rgb(255, 255, 255)',
                dark: 'rgb(50, 50, 50)',
            },
        },
    } satisfies Preset.ThemeConfig,

    themes: [
        '@docusaurus/theme-mermaid',
        [
            require.resolve('@easyops-cn/docusaurus-search-local'),
            {
                hashed: true,
                highlightSearchTermsOnTargetPage: true,
                searchBarShortcutHint: false,
            },
        ],
    ],
    stylesheets: [
        {
            href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
            type: 'text/css',
            integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
            crossorigin: 'anonymous',
        },
    ],
    plugins: [
        ScalarDocusaurus,
        [
            InstallGetiScript,
            {
                id: 'geti-installer',
                getiInstallerUrl: GETI_INSTALLER_LOCATION,
            },
        ],
        [
            '@docusaurus/plugin-sitemap',
            {
                id: 'intel-geti-sitemap',
                ignorePatterns: ['/search'],
            },
        ],
        [
            LLMsTxt,
            {
                siteDescription: DESCRIPTION,
                sidebarsConfig: SIDEBARS_CONFIG,
                shouldExportFile(item): boolean {
                    return item.type === 'link' && !item.file?.endsWith('redirect.md');
                },
            },
        ],
        'docusaurus-plugin-image-zoom',
    ],
};

export default config;
