/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Flex, View } from '@geti/ui';

import styles from './project-activity-status.module.scss';

interface ProjectActivityStatusProps {
    isActive: boolean;
}

const ProjectActivityStatusWrapper = ({ children }: { children: ReactNode }) => {
    return (
        <View
            backgroundColor={'gray-75'}
            borderWidth={'thin'}
            borderColor={'transparent'}
            borderRadius={'regular'}
            padding={'size-50'}
        >
            {children}
        </View>
    );
};

export const ProjectActivityStatus = ({ isActive }: ProjectActivityStatusProps) => {
    if (isActive) {
        return (
            <ProjectActivityStatusWrapper>
                <Flex alignItems={'center'} gap={'size-50'}>
                    <View UNSAFE_className={styles.activityStatusIndicator} />
                    <span className={styles.activityStatusText} aria-label={'Active project'}>
                        Active
                    </span>
                </Flex>
            </ProjectActivityStatusWrapper>
        );
    }

    return (
        <ProjectActivityStatusWrapper>
            <Flex>
                <span className={styles.activityStatusText} aria-label={'Inactive project'}>
                    Inactive
                </span>
            </Flex>
        </ProjectActivityStatusWrapper>
    );
};
