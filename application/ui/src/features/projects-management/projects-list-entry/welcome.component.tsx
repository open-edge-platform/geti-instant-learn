/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Fireworks } from '@geti-prompt/icons';
import { Button, Content, Flex, Heading, IllustratedMessage, Text } from '@geti/ui';

import { useCreateProject } from '../hooks/use-create-project.hook';
import { Layout } from './layout.component';

export const Welcome = () => {
    const createProject = useCreateProject();

    const handleCreateProject = () => {
        createProject('Project #1');
    };

    return (
        <Layout>
            <IllustratedMessage>
                <Fireworks />
                <Heading level={1}>Welcome to Geti Prompt!</Heading>
                <Content>
                    <Flex direction={'column'} gap={'size-200'}>
                        <Text>To start exploring visual and text prompts</Text>
                        <Button onPress={handleCreateProject}>Create project</Button>
                    </Flex>
                </Content>
            </IllustratedMessage>
        </Layout>
    );
};
