/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@geti-prompt/api';
import { Fireworks } from '@geti-prompt/icons';
import { Button, Content, Flex, Heading, IllustratedMessage, Text } from '@geti/ui';
import { Navigate, useNavigate } from 'react-router';
import { v4 as uuid } from 'uuid';

import { paths } from '../../../constants/paths';
import { useCreateProjectMutation } from '../hooks/use-create-project-mutation.hook';
import { Layout } from './layout.component';

const useCreateProject = () => {
    const createProjectMutation = useCreateProjectMutation();
    const navigate = useNavigate();

    const createProject = (projectName: string) => {
        const projectId = uuid();

        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    name: projectName,
                },
            },
            {
                onSuccess: () => {
                    navigate(paths.project({ projectId }));
                },
            }
        );
    };

    return createProject;
};

export const Welcome = () => {
    const createProject = useCreateProject();
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');

    if (data.projects.length > 0) {
        return <Navigate to={paths.root({})} replace />;
    }

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
