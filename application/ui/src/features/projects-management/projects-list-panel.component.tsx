/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api, type ProjectType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
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

import { useCreateProject } from './hooks/use-create-project.hook';
import { ActivateProjectDialog } from './project-activity/activate-project-dialog.component';
import { ProjectsList } from './projects-list.component';
import { ProjectWithActiveStatus } from './type';
import { generateUniqueProjectName } from './utils';

import styles from './projects-list.module.scss';

interface SelectedProjectProps {
    project: ProjectType;
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

interface CurrentProjectCardProps {
    selectedProject: ProjectWithActiveStatus;
    activeProject: ProjectType | undefined;
}

const useProjectActivityManagement = (projectId: string) => {
    const [isProjectActiveDialogOpen, setIsProjectActiveDialogOpen] = useState<boolean>(false);

    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects'],
                ['get', '/api/v1/projects/active'],
                ['get', '/api/v1/projects/{project_id}', { params: { path: { project_id: projectId } } }],
            ],
        },
        onSuccess: () => {
            handleCloseProjectActiveDialog();
        },
    });

    const handleCloseProjectActiveDialog = () => {
        setIsProjectActiveDialogOpen(false);
    };

    const handleUpdateProjectActivityStatus = (isGoingToBeActive: boolean) => {
        updateProjectMutation.mutate({
            body: {
                active: isGoingToBeActive,
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const handleDeactivateProject = () => {
        handleUpdateProjectActivityStatus(false);
    };

    const handleActivateProject = () => {
        handleUpdateProjectActivityStatus(true);
    };

    const handleShowActivateProjectDialog = () => {
        setIsProjectActiveDialogOpen(true);
    };

    return {
        isVisible: isProjectActiveDialogOpen,
        onClose: handleCloseProjectActiveDialog,
        onDeactivate: handleDeactivateProject,
        onActivate: handleActivateProject,
        onShowActivateProjectDialog: handleShowActivateProjectDialog,
        isPending: updateProjectMutation.isPending,
    };
};

const CurrentProjectCard = ({ selectedProject, activeProject }: CurrentProjectCardProps) => {
    const updateProjectMutation = $api.useMutation('put', '/api/v1/projects/{project_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects'],
                ['get', '/api/v1/projects/active'],
                ['get', '/api/v1/projects/{project_id}', { params: { path: { project_id: selectedProject.id } } }],
            ],
        },
    });

    const [isProjectActiveDialogOpen, setIsProjectActiveDialogOpen] = useState<boolean>(false);

    const handleCloseProjectActiveDialog = () => {
        setIsProjectActiveDialogOpen(false);
    };

    const handleUpdateProjectActiveStatus = () => {
        updateProjectMutation.mutate(
            {
                body: {
                    active: !selectedProject.isActive,
                },
                params: {
                    path: {
                        project_id: selectedProject.id,
                    },
                },
            },
            {
                onSuccess: () => {
                    handleCloseProjectActiveDialog();
                },
            }
        );
    };

    const handleClick = () => {
        if (!selectedProject.isActive && activeProject !== undefined) {
            setIsProjectActiveDialogOpen(true);

            return;
        }

        handleUpdateProjectActiveStatus();
    };

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

                    <Button variant={'primary'} onPress={handleClick} isPending={updateProjectMutation.isPending}>
                        {activeProject === undefined || !selectedProject.isActive ? 'Activate' : 'Deactivate'}
                    </Button>
                </Flex>
            </Header>
            <ActivateProjectDialog
                isVisible={isProjectActiveDialogOpen}
                onClose={handleCloseProjectActiveDialog}
                activeProjectName={activeProject?.name ?? ''}
                inactiveProjectName={selectedProject.name}
                onActivate={handleUpdateProjectActiveStatus}
            />
        </>
    );
};

export const ProjectsListPanel = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');
    const { data: currentProject } = $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });
    const { data: activeProjectResponse, isError } = $api.useQuery('get', '/api/v1/projects/active', undefined, {
        retry: false,
    });
    const [projectInEdition, setProjectInEdition] = useState<string | null>(null);

    const activeProject = isError ? undefined : activeProjectResponse;
    const selectedProject = { ...currentProject, isActive: activeProject?.id === currentProject.id };

    const projectsNames = data.projects.map((project) => project.name);
    const projects = data.projects.map((project) => ({
        ...project,
        isActive: project.id === activeProject?.id,
    }));

    return (
        <>
            <DialogTrigger type='popover' hideArrow>
                <SelectedProjectButton project={selectedProject} />

                <Dialog width={'size-4600'} UNSAFE_className={styles.dialog}>
                    <CurrentProjectCard selectedProject={selectedProject} activeProject={activeProject} />

                    <Content UNSAFE_className={styles.dialogContent}>
                        <Divider size={'S'} marginY={'size-200'} />

                        <ProjectsList
                            projects={projects}
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
