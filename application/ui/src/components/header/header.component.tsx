/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Flex, View } from '@geti/ui';
import { Link } from 'react-router-dom';

export const Header = ({ homeLink, children }: { homeLink: string; children: ReactNode }) => {
    return (
        <View gridArea={'header'} backgroundColor={'gray-200'}>
            <Flex height='100%' alignItems={'center'} justifyContent={'space-between'} marginX='1rem' gap='size-200'>
                <Link to={homeLink}>Geti Prompt</Link>
                {children}
            </Flex>
        </View>
    );
};
