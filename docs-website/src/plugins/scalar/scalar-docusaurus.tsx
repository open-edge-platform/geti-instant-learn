import { ApiReferenceReact } from '@scalar/api-reference-react';

import '@scalar/api-reference-react/style.css';
import '@scalar/docusaurus/dist/theme.css';

import React from 'react';

import Layout from '@theme/Layout';

import spec from './geti_openapi.json';

export const ScalarDocusaurus = () => {
    return (
        <Layout>
            <ApiReferenceReact
                configuration={{
                    spec: {
                        content: spec,
                    },
                    layout: 'modern',
                    showSidebar: true,
                    _integration: 'docusaurus',
                    hideModels: true,
                    forceDarkModeState: 'light',
                    hideClientButton: true,
                    hideDarkModeToggle: true,
                    metaData: {
                        title: 'REST API specification | Geti™',
                    },
                }}
            />
        </Layout>
    );
};

export default ScalarDocusaurus;
