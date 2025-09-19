/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import {
    ActionButton,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Header,
    Heading,
    PhotoPlaceholder,
    Text,
    View,
} from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { v4 as uuid } from 'uuid';

import { ProjectsList } from './projects-list.component';

import styles from './projects-list.module.scss';

interface SelectedProjectProps {
    name: string;
}

const SelectedProjectButton = ({ name }: SelectedProjectProps) => {
    return (
        <ActionButton aria-label={`Selected project ${name}`} isQuiet height={'max-content'} staticColor='white'>
            <View margin={'size-50'}>{name}</View>
            <View margin='size-50'>
                <PhotoPlaceholder name={name} email='' height={'size-400'} width={'size-400'} />
            </View>
        </ActionButton>
    );
};

export const ProjectsListPanel = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery('get', '/api/v1/projects');
    const addProjectMutation = $api.useMutation('post', '/api/v1/projects');
    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}');
    const deleteProjectMutation = $api.useMutation('delete', '/api/v1/projects/{project_id}');

    const [projectInEdition, setProjectInEdition] = useState<string | null>(null);

    const selectedProjectName = data?.projects.find((project) => project.id === projectId)?.name || '';

    const addProject = () => {
        const newProjectId = uuid();
        const newProjectName = `Project #${(data?.projects.length || 0) + 1}`;

        addProjectMutation.mutate({
            body: {
                id: newProjectId,
                name: newProjectName,
            },
        });

        setProjectInEdition(newProjectId);
    };

    const updateProjectName = (id: string, name: string): void => {
        updateProjectMutation.mutate({
            body: {
                name,
            },
            params: {
                path: {
                    project_id: id,
                },
            },
        });
    };

    const deleteProject = (id: string): void => {
        deleteProjectMutation.mutate({
            params: {
                path: {
                    project_id: id,
                },
            },
        });
    };

    return (
        <DialogTrigger type='popover' hideArrow>
            <SelectedProjectButton name={selectedProjectName} />

            <Dialog width={'size-4600'} UNSAFE_className={styles.dialog}>
                <Header>
                    <Flex direction={'column'} justifyContent={'center'} width={'100%'} alignItems={'center'}>
                        <PhotoPlaceholder
                            name={selectedProjectName}
                            email=''
                            height={'size-1000'}
                            width={'size-1000'}
                        />
                        <Heading level={2} marginBottom={0}>
                            {selectedProjectName}
                        </Heading>
                    </Flex>
                </Header>
                <Content>
                    <Divider size={'S'} marginY={'size-200'} />
                    <ProjectsList
                        projects={data?.projects ?? []}
                        projectIdInEdition={projectInEdition}
                        setProjectInEdition={setProjectInEdition}
                        onDeleteProject={deleteProject}
                        onUpdateProjectName={updateProjectName}
                    />
                    <Divider size={'S'} marginY={'size-200'} />
                </Content>

                <ButtonGroup UNSAFE_className={styles.panelButtons}>
                    <ActionButton
                        isQuiet
                        width={'100%'}
                        marginStart={'size-100'}
                        marginEnd={'size-350'}
                        UNSAFE_className={styles.addProjectButton}
                        onPress={addProject}
                    >
                        <AddCircle />
                        <Text marginX='size-50'>Add project</Text>
                    </ActionButton>
                </ButtonGroup>
            </Dialog>
        </DialogTrigger>
    );
};
