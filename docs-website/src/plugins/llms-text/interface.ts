import type { SidebarItemCategory, SidebarItemLink } from '@docusaurus/plugin-content-docs/src/sidebars/types.js';

export type LLMsTxtOptions = {
    siteDescription: string;
    sidebarsConfig: Record<string, { title: string; description: string }>;
    withFull?: boolean;
    matcher?: (item: Item) => boolean;
    fullmatcher?: (item: Item) => boolean;
    shouldExportFile: (item: ItemResult) => boolean;
};

export type Item = SidebarItemLink | SidebarItemCategory;
export type ItemResult =
    | {
          type: 'category';
          label: string;
          description?: string;
          items: ItemResult[];
          link: unknown;
      }
    | {
          type: 'link';
          label: string;
          description: string;
          href: string;
          file: string | undefined;
          metadata: {
              title: string;
              description: string;
              sidebar: string;
          };
      };
