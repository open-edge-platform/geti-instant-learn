/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api, type ProjectType } from '@/api';
import { useCurrentProject } from '@/hooks';
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

import { useCreateProject, useCreateProjectMutation } from './api/use-create-project.hook';
import { ProjectsList } from './projects-list.component';
import { generateUniqueProjectName } from './utils';

import styles from './projects-list.module.scss';

interface SelectedProjectProps {
    project: ProjectType;
}

const SelectedProjectButton = ({ project: { name, id, active } }: SelectedProjectProps) => {
    return (
        <ActionButton aria-label={`Selected project ${name}`} isQuiet height={'max-content'} data-active={active}>
            <View margin={'size-50'}>
                <Flex direction={'column'} gap={'size-50'}>
                    <Text UNSAFE_className={styles.currentProjectHeaderText}>{name}</Text>
                </Flex>
            </View>
            <View margin='size-50'>
                <PhotoPlaceholder name={name} indicator={id} height={'size-400'} width={'size-400'} />
            </View>
        </ActionButton>
    );
};

interface AddProjectProps {
    projectsNames: string[];
}

const CreateProjectButton = ({ projectsNames }: AddProjectProps) => {
    const createProjectMutation = useCreateProjectMutation();
    const createProject = useCreateProject();

    const handleCreateProject = () => {
        const name = generateUniqueProjectName(projectsNames);

        createProject.mutate({ name });
    };

    return (
        <>
            <ActionButton
                isQuiet
                width={'100%'}
                marginStart={'size-100'}
                marginEnd={'size-350'}
                UNSAFE_className={styles.createProjectButton}
                isDisabled={createProjectMutation.isPending}
                onPress={handleCreateProject}
            >
                <AddCircle />
                <Text marginX='size-50'>Create project</Text>
            </ActionButton>
        </>
    );
};

interface CurrentProjectCardProps {
    selectedProject: ProjectType;
}

const CurrentProjectCard = ({ selectedProject }: CurrentProjectCardProps) => {
    return (
        <Header>
            <Flex
                direction={'column'}
                justifyContent={'center'}
                width={'100%'}
                alignItems={'center'}
                UNSAFE_className={styles.currentProject}
                gap={'size-200'}
            >
                <Flex alignItems={'center'} direction={'column'}>
                    <PhotoPlaceholder
                        name={selectedProject.name}
                        indicator={selectedProject.id}
                        height={'size-1000'}
                        width={'size-1000'}
                    />

                    <Heading level={2} marginBottom={0}>
                        {selectedProject.name}
                    </Heading>
                </Flex>
            </Flex>
        </Header>
    );
};

export const ProjectsListPanel = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');
    const { data: currentProject } = useCurrentProject();
    const [projectInEdition, setProjectInEdition] = useState<string | null>(null);

    const projectsNames = data.projects.map((project) => project.name);

    return (
        <>
            <DialogTrigger type='popover' hideArrow>
                <SelectedProjectButton project={currentProject} />

                <Dialog width={'size-4600'} UNSAFE_className={styles.dialog}>
                    <CurrentProjectCard selectedProject={currentProject} />

                    <Content UNSAFE_className={styles.dialogContent}>
                        <Divider size={'S'} marginY={'size-200'} />

                        <ProjectsList
                            projects={data.projects}
                            projectIdInEdition={projectInEdition}
                            setProjectInEdition={setProjectInEdition}
                        />
                        <Divider size={'S'} marginY={'size-200'} />
                    </Content>

                    <ButtonGroup UNSAFE_className={styles.panelButtons}>
                        <CreateProjectButton projectsNames={projectsNames} />
                    </ButtonGroup>
                </Dialog>
            </DialogTrigger>
        </>
    );
};
