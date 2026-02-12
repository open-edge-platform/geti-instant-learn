import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';
import type { PluginRouteConfig } from '@docusaurus/types';

import { Item, ItemResult } from './interface';

function mapPathsToSourceFilePath(routes: PluginRouteConfig[], baseUrl: string) {
    const docsPluginRouteConfig = routes.find((route) => route.plugin.name === 'docusaurus-plugin-content-docs');
    const docsRoutes = docsPluginRouteConfig.routes?.find((route) => route.path === `${baseUrl}docs`);

    const docs = docsRoutes['props']['version']['docs'];
    const formattedDocs = Object.keys(docs).map((key) => {
        const doc = docs[key];
        return {
            id: doc.id as string,
            title: doc.title as string,
            description: doc.description as string,
            sidebar: doc.sidebar as string,
        };
    });

    const files = docsRoutes.routes.flatMap((docRoot) => {
        return docRoot.routes
            .map((route) => {
                if (!route.metadata) {
                    console.log('Could not get metadata', route);
                    return undefined;
                }
                return {
                    path: route.path,
                    file: route.metadata?.sourceFilePath,
                    sidebar: route.sidebar as string | undefined,
                };
            })
            .filter((x) => x !== undefined);
    });

    const pathToFile = Object.fromEntries(files.map(({ path, file }) => [path, file]));
    const idToDescriptionSidebarAndTItle = Object.fromEntries(
        formattedDocs.map(({ id, title, description, sidebar }) => [id, { title, description, sidebar }])
    );

    const getFilePath = (item: Item) => ('href' in item ? pathToFile[item.href] : undefined);
    const getMetadata = (item: Item) =>
        'docId' in item && typeof item.docId === 'string' ? idToDescriptionSidebarAndTItle[item.docId] : undefined;

    return { getFilePath, getMetadata };
}

export function getDocItemsFromSidebars(
    routes: PluginRouteConfig[],
    baseUrl: string
): { key: string; items: ItemResult[] }[] {
    const { getFilePath, getMetadata } = mapPathsToSourceFilePath(routes, baseUrl);

    // we need to dig down several layers:
    // find PluginRouteConfig marked by plugin.name === "docusaurus-plugin-content-docs"
    const docsPluginRouteConfig = routes.find((route) => route.plugin.name === 'docusaurus-plugin-content-docs');

    const version = docsPluginRouteConfig.routes?.find((route) => route.path === `${baseUrl}docs`).props
        .version as Record<string, unknown>;

    function parseItem(item: Item): ItemResult[] {
        if (item.type === 'link') {
            const file = getFilePath(item);
            const metadata = getMetadata(item);

            return [
                {
                    type: 'link',
                    label: item.label,
                    href: item.href,
                    description: item.description,
                    file,
                    metadata,
                    ...item,
                },
            ];
        }

        if (item.type === 'category') {
            return [
                {
                    type: 'category',
                    label: item.label,
                    link: item.link,
                    description: item.description,
                    items: item.items.flatMap(parseItem),
                },
            ];
        }

        return [];
    }

    const docSidebars = version.docsSidebars as SidebarsConfig;
    return Object.keys(docSidebars).map((key) => {
        const sidebar = docSidebars[key] as Item[];

        const parsedItems = sidebar.flatMap((item: Item) => {
            if (typeof item === 'string') {
                return undefined;
            }

            return parseItem(item as Item);
        });
        return { key, items: parsedItems };
    });
}
