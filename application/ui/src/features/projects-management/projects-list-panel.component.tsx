/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api, type ProjectListItemType } from '@geti-prompt/api';
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

import { useCreateProject } from './hooks/use-create-project.hook';
import { ProjectsList } from './projects-list.component';
import { generateUniqueProjectName } from './utils';

import styles from './projects-list.module.scss';

interface SelectedProjectProps {
    project: ProjectListItemType;
}

const SelectedProjectButton = ({ project: { name, id } }: SelectedProjectProps) => {
    return (
        <ActionButton aria-label={`Selected project ${name}`} isQuiet height={'max-content'} staticColor='white'>
            <View margin={'size-50'}>{name}</View>
            <View margin='size-50'>
                <PhotoPlaceholder name={name} indicator={id} height={'size-400'} width={'size-400'} />
            </View>
        </ActionButton>
    );
};

interface AddProjectProps {
    onSetProjectInEdition: (projectId: string) => void;
    projectsNames: string[];
}

const CreateProjectButton = ({ onSetProjectInEdition, projectsNames }: AddProjectProps) => {
    const createProject = useCreateProject();

    const handleCreateProject = () => {
        const newProjectId = uuid();
        const newProjectName = generateUniqueProjectName(projectsNames);

        createProject(newProjectName, newProjectId);

        onSetProjectInEdition(newProjectId);
    };

    return (
        <ActionButton
            isQuiet
            width={'100%'}
            marginStart={'size-100'}
            marginEnd={'size-350'}
            UNSAFE_className={styles.createProjectButton}
            onPress={handleCreateProject}
        >
            <AddCircle />
            <Text marginX='size-50'>Create project</Text>
        </ActionButton>
    );
};

export const ProjectsListPanel = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');

    const [projectInEdition, setProjectInEdition] = useState<string | null>(null);
    const selectedProject = data.projects.find((project) => project.id === projectId);

    if (!selectedProject) {
        return <div>No project found</div>;
    }

    const projectsNames = data.projects.map((project) => project.name);

    return (
        <DialogTrigger type='popover' hideArrow>
            <SelectedProjectButton project={selectedProject} />

            <Dialog width={'size-4600'} UNSAFE_className={styles.dialog}>
                <Header>
                    <Flex direction={'column'} justifyContent={'center'} width={'100%'} alignItems={'center'}>
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
                </Header>
                <Content>
                    <Divider size={'S'} marginY={'size-200'} />
                    <ProjectsList
                        projects={data.projects}
                        projectIdInEdition={projectInEdition}
                        setProjectInEdition={setProjectInEdition}
                    />
                    <Divider size={'S'} marginY={'size-200'} />
                </Content>

                <ButtonGroup UNSAFE_className={styles.panelButtons}>
                    <CreateProjectButton onSetProjectInEdition={setProjectInEdition} projectsNames={projectsNames} />
                </ButtonGroup>
            </Dialog>
        </DialogTrigger>
    );
};
