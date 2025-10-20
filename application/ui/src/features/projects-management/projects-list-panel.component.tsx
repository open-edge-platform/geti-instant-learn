/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api, type ProjectType } from '@geti-prompt/api';
import {
    ActionButton,
    Button,
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

import { ActivateProjectDialog } from './activate-project-dialog/activate-project-dialog.component';
import { useCreateProject } from './hooks/use-create-project.hook';
import { useCurrentProject } from './hooks/use-current-project.hook';
import { useProjectActivityManagement } from './hooks/use-project-activity-management.hook';
import { ProjectActivityStatus } from './project-activity-status/project-activity-status.component';
import { ProjectsList } from './projects-list.component';
import { generateUniqueProjectName } from './utils';

import styles from './projects-list.module.scss';

interface SelectedProjectProps {
    project: ProjectType;
}

const SelectedProjectButton = ({ project: { name, id, active } }: SelectedProjectProps) => {
    return (
        <ActionButton aria-label={`Selected project ${name}`} isQuiet height={'max-content'}>
            <View margin={'size-50'}>
                <Flex direction={'column'} gap={'size-50'}>
                    <Text UNSAFE_className={styles.currentProjectHeaderText}>{name}</Text>
                    <View alignSelf={'end'}>
                        <ProjectActivityStatus isActive={active} />
                    </View>
                </Flex>
            </View>
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

interface CurrentProjectCardProps {
    selectedProject: ProjectType;
    activeProject: ProjectType | undefined;
}

const CurrentProjectCard = ({ selectedProject, activeProject }: CurrentProjectCardProps) => {
    const { isVisible, close, activate, deactivate, isPending, activateConfirmation } = useProjectActivityManagement(
        selectedProject.id,
        activeProject?.id
    );

    const handleClick = () => {
        if (selectedProject.active) {
            deactivate();
        } else {
            activate();
        }
    };

    const buttonText = activeProject === undefined || !selectedProject.active ? 'Activate' : 'Deactivate';

    return (
        <>
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

                    <Button
                        variant={'primary'}
                        onPress={handleClick}
                        isPending={isPending}
                        aria-label={`${buttonText} current project`}
                    >
                        {buttonText}
                    </Button>
                </Flex>
            </Header>
            <ActivateProjectDialog
                isVisible={isVisible}
                onClose={close}
                activeProjectName={activeProject?.name ?? ''}
                inactiveProjectName={selectedProject.name}
                onActivate={activateConfirmation}
            />
        </>
    );
};

export const ProjectsListPanel = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');
    const { data: currentProject } = useCurrentProject();
    const [projectInEdition, setProjectInEdition] = useState<string | null>(null);

    const activeProject = data.projects.find((project) => project.active);

    const projectsNames = data.projects.map((project) => project.name);

    return (
        <>
            <DialogTrigger type='popover' hideArrow>
                <SelectedProjectButton project={currentProject} />

                <Dialog width={'size-4600'} UNSAFE_className={styles.dialog}>
                    <CurrentProjectCard selectedProject={currentProject} activeProject={activeProject} />

                    <Content UNSAFE_className={styles.dialogContent}>
                        <Divider size={'S'} marginY={'size-200'} />

                        <ProjectsList
                            projects={data.projects}
                            activeProject={activeProject}
                            projectIdInEdition={projectInEdition}
                            setProjectInEdition={setProjectInEdition}
                        />
                        <Divider size={'S'} marginY={'size-200'} />
                    </Content>

                    <ButtonGroup UNSAFE_className={styles.panelButtons}>
                        <CreateProjectButton
                            onSetProjectInEdition={setProjectInEdition}
                            projectsNames={projectsNames}
                        />
                    </ButtonGroup>
                </Dialog>
            </DialogTrigger>
        </>
    );
};
