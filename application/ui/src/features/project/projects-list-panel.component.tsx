/**
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, type ProjectType } from '@/api';
import { ProjectActions } from '@/features/project/project-list-item/project-actions.component';
import { useCurrentProject } from '@/hooks';
import {
    ActionButton,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    dimensionValue,
    Divider,
    Flex,
    Header,
    Heading,
    PhotoPlaceholder,
    Text,
    View,
} from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';
import { useNavigate } from 'react-router';

import { paths } from '../../constants/paths';
import { useCreateProject, useCreateProjectMutation } from './api/use-create-project.hook';
import { ProjectsList } from './projects-list.component';
import { generateUniqueProjectName } from './utils';

import classes from './projects-list.module.scss';

interface SelectedProjectProps {
    project: ProjectType;
}

const SelectedProjectButton = ({ project: { name, id, active } }: SelectedProjectProps) => {
    return (
        <ActionButton
            aria-label={`Selected project ${name}`}
            isQuiet
            height={'max-content'}
            data-active={active}
            UNSAFE_className={classes.selectedProjectButton}
        >
            <View margin={'size-50'}>
                <Flex direction={'column'} gap={'size-50'}>
                    <Text UNSAFE_className={classes.currentProjectHeaderText}>{name}</Text>
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
                UNSAFE_className={classes.createProjectButton}
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
    projectNames: string[];
}

const CurrentProjectCard = ({ selectedProject, projectNames }: CurrentProjectCardProps) => {
    const navigate = useNavigate();

    const handleNavigateAfterDeletion = () => {
        if (projectNames.length === 0) {
            navigate(paths.welcome({}));
        } else {
            navigate(paths.projects({}));
        }
    };

    return (
        <Header>
            <Flex
                direction={'column'}
                justifyContent={'center'}
                width={'100%'}
                alignItems={'center'}
                UNSAFE_className={classes.currentProject}
                gap={'size-200'}
            >
                <Flex alignItems={'center'} direction={'column'} width={'100%'}>
                    <PhotoPlaceholder
                        name={selectedProject.name}
                        indicator={selectedProject.id}
                        height={'size-1000'}
                        width={'size-1000'}
                    />

                    <View position={'relative'} width={'100%'} marginTop={'size-225'}>
                        <Flex alignItems={'center'} justifyContent={'center'}>
                            <Heading level={2} margin={0} marginX={'size-500'} UNSAFE_className={classes.projectTitle}>
                                {selectedProject.name}
                            </Heading>
                        </Flex>

                        <ProjectActions
                            projectId={selectedProject.id}
                            projectName={selectedProject.name}
                            projectNames={projectNames}
                            onDeleted={handleNavigateAfterDeletion}
                            actionButtonStyle={{
                                position: 'absolute',
                                top: '50%',
                                right: dimensionValue('size-100'),
                                transform: 'translateY(-50%)',
                            }}
                        />
                    </View>
                </Flex>
            </Flex>
        </Header>
    );
};

export const ProjectsListPanel = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');
    const { data: currentProject } = useCurrentProject();

    const projectsNames = data.projects.map((project) => project.name);
    const restProjects = data.projects.filter((project) => project.id !== currentProject.id);

    return (
        <>
            <DialogTrigger type='popover' hideArrow>
                <SelectedProjectButton project={currentProject} />

                <Dialog width={'size-4600'} UNSAFE_className={classes.dialog}>
                    <CurrentProjectCard
                        selectedProject={currentProject}
                        projectNames={projectsNames.filter((projectName) => projectName !== currentProject.name)}
                    />

                    {!isEmpty(restProjects) && (
                        <>
                            <Divider size={'S'} marginBottom={'size-100'} marginTop={0} />
                            <Content>
                                <ProjectsList projects={restProjects} />
                            </Content>
                        </>
                    )}

                    <ButtonGroup UNSAFE_className={classes.buttonsGroup}>
                        <CreateProjectButton projectsNames={projectsNames} />
                    </ButtonGroup>
                </Dialog>
            </DialogTrigger>
        </>
    );
};
