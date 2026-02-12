import fs from 'node:fs';
import path from 'node:path';

import { ItemResult } from './interface';

type Options = { siteDir: string; outDir: string; shouldExportFile: (item: ItemResult) => boolean };
export async function copyOverMarkdownFiles(
    docItems: { key: string; items: ItemResult[] }[],
    { siteDir, outDir, shouldExportFile }: Options
) {
    const copyFile = async (item: ItemResult) => {
        if ('file' in item && item.file !== undefined) {
            if (!shouldExportFile(item)) {
                return;
            }

            try {
                const content = fs.readFileSync(path.join(siteDir, item.file));
                const filePath = path.join(outDir, item.file);

                await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
                fs.writeFileSync(filePath, content, { encoding: 'utf8' });
            } catch (e) {
                console.log(`Error when saving markdown file`, path.join(outDir, item.file));
                console.error(e);
            }
        }

        if (item.type === 'category') {
            for (const subItem of item.items) {
                await copyFile(subItem);
            }
        }
    };

    const items = docItems.flatMap((sidebar) => sidebar.items);
    for (const item of items) {
        await copyFile(item);
    }
}
