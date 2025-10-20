/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, MouseEventHandler, useState } from 'react';

import { type ProjectType } from '@geti-prompt/api';
import { Flex, PhotoPlaceholder, Text } from '@geti/ui';
import { Link } from 'react-router-dom';

import { paths } from '../../../routes/paths';
import { ActivateProjectDialog } from '../activate-project-dialog/activate-project-dialog.component';
import { useProjectActivityManagement } from '../hooks/use-project-activity-management.hook';
import { DeleteProjectDialog, PROJECT_ACTIONS, ProjectActions, ProjectEdition } from './project-actions.component';

import styles from './project-list-item.module.scss';

interface ProjectListItemProps {
    project: ProjectType;
    activeProject: ProjectType | undefined;
    isInEditMode: boolean;
    onBlur: (projectId: string, newName: string) => void;
    onRename: (projectId: string) => void;
    onDelete: (projectId: string) => void;
    onResetProjectInEdition: () => void;
    projectNames: string[];
}

export const ProjectListItem = ({
    project,
    isInEditMode,
    onBlur,
    onRename,
    onDelete,
    onResetProjectInEdition,
    projectNames,
    activeProject,
}: ProjectListItemProps) => {
    const { isVisible, close, activate, deactivate, activateConfirmation } = useProjectActivityManagement(
        project.id,
        activeProject?.id
    );

    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);

    const handleAction = (key: Key) => {
        if (key === PROJECT_ACTIONS.RENAME) {
            onRename(project.id);
        } else if (key === PROJECT_ACTIONS.DELETE) {
            setIsDeleteDialogOpen(true);
        } else if (key === PROJECT_ACTIONS.ACTIVATE) {
            activate();
        } else if (key === PROJECT_ACTIONS.DEACTIVATE) {
            deactivate();
        }
    };

    const handleBlur = (projectId: string) => (newName: string) => {
        onBlur(projectId, newName);
    };

    const handleDelete = () => {
        onDelete(project.id);
    };

    const handleItemClick: MouseEventHandler<HTMLAnchorElement> = (event) => {
        if (isInEditMode) {
            event.preventDefault();
        }
    };

    const projectActions = [
        PROJECT_ACTIONS.RENAME,
        PROJECT_ACTIONS.DELETE,
        project.active ? PROJECT_ACTIONS.DEACTIVATE : PROJECT_ACTIONS.ACTIVATE,
    ];

    return (
        <li className={styles.projectListItem} aria-label={`Project ${project.name}`}>
            <Link to={paths.project({ projectId: project.id })} onClick={handleItemClick}>
                <Flex justifyContent='space-between' alignItems='center'>
                    {isInEditMode ? (
                        <ProjectEdition
                            name={project.name}
                            onBlur={handleBlur(project.id)}
                            onResetProjectInEdition={onResetProjectInEdition}
                            projectNames={projectNames}
                        />
                    ) : (
                        <Flex alignItems={'center'} gap={'size-100'}>
                            <PhotoPlaceholder
                                name={project.name}
                                indicator={project.id}
                                height={'size-300'}
                                width={'size-300'}
                            />
                            <Text>{project.name}</Text>
                        </Flex>
                    )}
                    <ProjectActions actions={projectActions} onAction={handleAction} />
                </Flex>
            </Link>
            <DeleteProjectDialog
                isOpen={isDeleteDialogOpen}
                onDismiss={() => setIsDeleteDialogOpen(false)}
                onDelete={handleDelete}
                projectName={project.name}
            />
            {/*
                Activate Project Dialog is only visible when there is already an active project.
                When there is no active project, the dialog is not visible; we just activate the selected project
                directly.
            */}
            <ActivateProjectDialog
                isVisible={isVisible}
                onClose={close}
                activeProjectName={activeProject?.name ?? ''}
                inactiveProjectName={project.name}
                onActivate={activateConfirmation}
            />
        </li>
    );
};
