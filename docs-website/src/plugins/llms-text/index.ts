import fs from 'node:fs';
import path from 'node:path';

import type { LoadContext, Plugin } from '@docusaurus/types';

import { buildLLMsTxt } from './build-llms-txt';
import { copyOverMarkdownFiles } from './copy-over-markdown-files';
import { getDocItemsFromSidebars } from './get-doc-items-from-sidebars';
import { LLMsTxtOptions } from './interface';

export function LLMsTxt(_context: LoadContext, options: LLMsTxtOptions): Plugin {
    return {
        name: 'llms-txt-plugin',

        postBuild: async ({ routes, outDir, siteConfig, siteDir }) => {
            console.log('Generating llms.txt');

            const sidebars = getDocItemsFromSidebars(routes, siteConfig.baseUrl);
            const config = { title: siteConfig.title, siteUrl: siteConfig.url, siteDir, outDir, ...options };

            console.log('Building llms.txt');

            // Write llms.txt file
            const llmsTxt = buildLLMsTxt(sidebars, config);
            fs.writeFileSync(path.join(outDir, 'llms.txt'), llmsTxt);

            console.log('Building llms-full.txt');

            // Write llms-full.txt file
            const llmsTxtFull = buildLLMsTxt(sidebars, config, true);
            fs.writeFileSync(path.join(outDir, 'llms-full.txt'), llmsTxtFull);

            console.log('Copying markdown output');
            await copyOverMarkdownFiles(sidebars, config);
            console.log('Finished copying markdown output');
        },
    };
}
