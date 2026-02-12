import fs from 'node:fs';
import path from 'node:path';

import { ItemResult, LLMsTxtOptions } from './interface';

function formatItem(
    item: ItemResult,
    depth: number,
    {
        siteDir,
        siteUrl,
        outDir,
        shouldExportFile,
    }: {
        siteDir: string;
        siteUrl: string;
        outDir: string;
        shouldExportFile: (item: ItemResult) => boolean;
    },
    full = false
): string[] {
    if (item.type === 'category') {
        return [
            ``,
            `${'#'.repeat(depth)} ${item.label}`,
            ...(item.description ? ['', item.description] : []),
            '',
            ...item.items.flatMap((i) =>
                formatItem(i, depth + 1, { siteDir, siteUrl, outDir, shouldExportFile }, full)
            ),
            //``,
        ];
    }

    if (item.type === 'link') {
        if (item.file?.endsWith('redirect.md')) {
            item.file = undefined;
        }

        const url = `${siteUrl}${item.href}`;
        const fileUrl = item.file ? `${siteUrl}/${item.file}` : undefined;

        const title = item.metadata ? item.metadata.title : item.label;
        const description = item.metadata?.description ?? item.description;

        if (full && item.file) {
            const content = fs.readFileSync(path.join(siteDir, item.file)).toString();

            return [content];
        }

        if (shouldExportFile(item)) {
            return [`- [${title}](${fileUrl}): ${description}`];
        } else {
            return [`- [${title}](${url}): ${description}`];
        }
    }

    return [];
}

type Options = {
    siteDir: string;
    siteUrl: string;
    title: string;
    outDir: string;
    siteDescription: string;
    sidebarsConfig: LLMsTxtOptions['sidebarsConfig'];
    shouldExportFile: (item: ItemResult) => boolean;
};

export function buildLLMsTxt(sidebars: { key: string; items: ItemResult[] }[], options: Options, full = false) {
    const docsRecords = sidebars
        .flatMap((sidebar) => {
            const sidebarDetails = options.sidebarsConfig[sidebar.key];

            // Content of every sidebar item
            const content = sidebar.items.map((i) => formatItem(i, 3, options, full)).flatMap((item) => item);

            if (sidebarDetails === undefined) {
                return content;
            }

            // Add title and descriptoin of current sidebar
            return [`## ${sidebarDetails.title}`, ``, sidebarDetails.description, ``, ...content, ``];
        })
        .join('\n');

    // Build up llms.txt file
    const llmsTxt = `# ${options.title}

${options.siteDescription}


${docsRecords}`;

    return llmsTxt;
}
