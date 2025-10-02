/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { Fireworks } from '@geti-prompt/icons';
import { Button, Content, Flex, Heading, IllustratedMessage, Text, View } from '@geti/ui';
import { useNavigate } from 'react-router';
import { v4 as uuid } from 'uuid';

import { paths } from '../../../routes/paths';
import { Layout } from './layout.component';

export const Welcome = () => {
    const createProjectMutation = $api.useMutation('post', '/api/v1/projects');
    const navigate = useNavigate();

    const createProject = () => {
        const projectId = uuid();

        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    name: 'Project #1',
                },
            },
            {
                onSuccess: () => {
                    navigate(paths.project({ projectId }));
                },
            }
        );
    };

    return (
        <Layout>
            <IllustratedMessage>
                <Fireworks />
                <Heading level={1}>Welcome to Geti Prompt!</Heading>
                <Content>
                    <Flex direction={'column'} gap={'size-200'}>
                        <Text>To start exploring visual and text prompts</Text>
                        <Button onPress={createProject}>Create project</Button>
                    </Flex>
                </Content>
            </IllustratedMessage>
        </Layout>
    );
};
