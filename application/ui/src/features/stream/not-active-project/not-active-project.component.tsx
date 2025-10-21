/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api, ProjectType } from '@geti-prompt/api';
import { NoActiveProject as NoActiveProjectIcon } from '@geti-prompt/icons';
import { Button, Flex, Text, View } from '@geti/ui';

import { ActivateProjectDialog } from '../../projects-management/activate-project-dialog/activate-project-dialog.component';
import { useProjectActivityManagement } from '../../projects-management/hooks/use-project-activity-management.hook';

import styles from './not-active-project.module.scss';

interface NotActiveProjectProps {
    project: ProjectType;
}

export const NotActiveProject = ({ project }: NotActiveProjectProps) => {
    const { data } = $api.useSuspenseQuery('get', '/api/v1/projects');
    const activeProject = data.projects.find(({ active }) => active);

    const { isVisible, activate, close, isPending, activateConfirmation } = useProjectActivityManagement(
        project.id,
        activeProject?.id
    );

    return (
        <>
            <View height={'100%'} paddingX={'size-800'} paddingY={'size-1000'}>
                <View backgroundColor={'gray-200'} height={'100%'} UNSAFE_className={styles.container}>
                    <Flex
                        alignItems={'center'}
                        justifyContent={'center'}
                        direction={'column'}
                        height={'100%'}
                        gap={'size-200'}
                    >
                        <NoActiveProjectIcon />
                        <Flex direction={'column'} alignItems={'center'} width={'40rem'} gap={'size-100'}>
                            <Text UNSAFE_className={styles.inactiveDescription}>
                                This project is set as inactive, therefore the pipeline configuration is disabled for
                                this project. You can still explore the sources configuration within this inactive
                                project.
                            </Text>
                            <Text UNSAFE_className={styles.activateQuestion}>
                                Would you like to activate this project?
                            </Text>
                        </Flex>

                        <Button
                            variant={'primary'}
                            isPending={isPending}
                            onPress={activate}
                            aria-label={'Activate current project'}
                        >
                            Activate project
                        </Button>
                    </Flex>
                </View>
            </View>
            <ActivateProjectDialog
                isVisible={isVisible}
                onClose={close}
                activeProjectName={activeProject?.name ?? ''}
                inactiveProjectName={project.name}
                onActivate={activateConfirmation}
            />
        </>
    );
};
