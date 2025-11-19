/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ProjectType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { isEmpty } from 'lodash-es';
import { useNavigate } from 'react-router';

import { paths } from '../../constants/paths';
import { useDeleteProject } from './hooks/use-delete-project.hook';
import { useUpdateProject } from './hooks/use-update-project.hook';
import { ProjectListItem } from './project-list-item/project-list-item.component';

import styles from './projects-list.module.scss';

interface ProjectListProps {
    projects: ProjectType[];
    projectIdInEdition: string | null;
    setProjectInEdition: (projectId: string | null) => void;
    activeProject: ProjectType | undefined;
}

export const ProjectsList = ({
    projects,
    activeProject,
    setProjectInEdition,
    projectIdInEdition,
}: ProjectListProps) => {
    const updateProjectName = useUpdateProject();
    const deleteProject = useDeleteProject();
    const { projectId } = useProjectIdentifier();
    const navigate = useNavigate();

    const projectNames = projects.map((project) => project.name);

    const handleDelete = (id: string): void => {
        deleteProject(id, () => {
            if (projects.length > 1 && id === projectId) {
                navigate(paths.projects({}));
            } else if (projects.length === 1) {
                navigate(paths.welcome({}));
            }
        });
    };

    const isInEditionMode = (id: string) => {
        return projectIdInEdition === id;
    };

    const handleBlur = (id: string, newName: string) => {
        const projectToUpdate = projects.find((project) => project.id === id);
        if (projectToUpdate?.name === newName || isEmpty(newName.trim())) {
            return;
        }

        updateProjectName(id, { name: newName });
    };

    const handleRename = (id: string) => {
        setProjectInEdition(id);
    };

    const handleResetProjectInEdition = () => {
        setProjectInEdition(null);
    };

    return (
        <ul className={styles.projectList}>
            {projects.map((project) => (
                <ProjectListItem
                    key={project.id}
                    activeProject={activeProject}
                    projectNames={projectNames.filter((name) => name !== project.name)}
                    project={project}
                    onRename={handleRename}
                    onDelete={handleDelete}
                    onBlur={handleBlur}
                    isInEditMode={isInEditionMode(project.id)}
                    onResetProjectInEdition={handleResetProjectInEdition}
                />
            ))}
        </ul>
    );
};
