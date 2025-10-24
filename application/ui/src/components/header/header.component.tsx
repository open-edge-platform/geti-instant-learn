/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, Header as SpectrumHeader, View } from '@geti/ui';

import { ProjectsListPanel } from '../../features/projects-management/projects-list-panel.component';

export const Header = () => {
    return (
        <View gridArea={'header'} backgroundColor={'gray-200'}>
            <Flex height='100%' alignItems={'center'} justifyContent={'space-between'} marginX='1rem' gap='size-200'>
                <SpectrumHeader>Geti Prompt</SpectrumHeader>
                <ProjectsListPanel />
            </Flex>
        </View>
    );
};
