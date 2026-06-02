/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ProjectType } from '@/api';
import { useProjectIdentifier } from '@/hooks';

import { useActivateProject } from './api/use-activate-project.hook';
import { ProjectListItem } from './project-list-item/project-list-item.component';

import classes from './projects-list.module.scss';

interface ProjectListProps {
    projects: ProjectType[];
}

export const ProjectsList = ({ projects }: ProjectListProps) => {
    const { projectId } = useProjectIdentifier();
    const activateProject = useActivateProject();

    const projectNames = projects.map((project) => project.name);

    const handleActivateProject = (project: ProjectType) => {
        if (project.active) return;

        const activeProject = projects.find(({ active }) => active);

        activateProject.mutate(project, activeProject);
    };

    return (
        <ul className={classes.projectList}>
            {projects
                .filter((project) => project.id !== projectId)
                .map((project) => (
                    <ProjectListItem
                        key={project.id}
                        projectNames={projectNames.filter((name) => name !== project.name)}
                        project={project}
                        onActivateProject={handleActivateProject}
                    />
                ))}
        </ul>
    );
};
