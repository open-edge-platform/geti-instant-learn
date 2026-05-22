/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Flex, View } from '@geti/ui';
import { Link } from 'react-router-dom';

import getiLogo from '../../assets/icons/geti-instant-learn-logo.webp';

import styles from './header.component.module.scss';

export const Header = ({ homeLink, children }: { homeLink: string; children: ReactNode }) => {
    return (
        <View gridArea={'header'} backgroundColor={'gray-200'}>
            <Flex height='100%' alignItems={'center'} justifyContent={'space-between'} marginX='1rem' gap='size-200'>
                <Link to={homeLink}>
                    <Flex alignItems='center' gap='size-50'>
                        <img src={getiLogo} alt={'Geti Instant Learn'} className={styles.logo} />
                        Geti™ Instant Learn
                    </Flex>
                </Link>
                {children}
            </Flex>
        </View>
    );
};
