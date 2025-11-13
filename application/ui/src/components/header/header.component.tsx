/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, View } from '@geti/ui';
import { Link } from 'react-router-dom';

import { ProjectsListPanel } from '../../features/project/projects-list-panel.component';
import { paths } from '../../routes/paths';

export const Header = () => {
    return (
        <View gridArea={'header'} backgroundColor={'gray-200'}>
            <Flex height='100%' alignItems={'center'} justifyContent={'space-between'} marginX='1rem' gap='size-200'>
                <Link to={paths.projects({})}>Geti Prompt</Link>
                <ProjectsListPanel />
            </Flex>
        </View>
    );
};
