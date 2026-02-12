import path from 'node:path';

import type { LoadContext, Plugin } from '@docusaurus/types';
import type { ReferenceProps } from '@scalar/api-reference-react';

/**
 * Scalar's Docusaurus plugin for Api References
 */
const ScalarDocusaurus = (context: LoadContext): Plugin<ReferenceProps> => {
    return {
        name: '@scalar/docusaurus',

        async contentLoaded({ content, actions }) {
            const { addRoute } = actions;

            addRoute({
                path: `${context.baseUrl}docs/rest-api/openapi-specification`,
                component: path.resolve(__dirname, './scalar-docusaurus'),
                exact: true,
                ...content,
            });
        },
    };
};

export default ScalarDocusaurus;
