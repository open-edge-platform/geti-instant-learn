/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { ActionMenu, Flex, Heading, Item, View } from '@geti/ui';
import { clsx } from 'clsx';

import styles from './source-read-only.module.scss';

interface SourceReadOnlyProps {
    isActive: boolean;
    parameters: string[];
    icon: ReactNode;
    title: string;
    menu?: ReactNode;
}

interface SourceMenuProps {
    isActive: boolean;
    onAction: (action: string) => void;
    items: { key: string; label: string }[];
}

const SourceMenu = ({ onAction, isActive, items }: SourceMenuProps) => {
    return (
        <ActionMenu
            isQuiet
            UNSAFE_className={clsx(styles.sourceMenu, {
                [styles.sourceActiveMenu]: isActive,
            })}
            onAction={(key) => onAction(String(key))}
            items={items}
        >
            {(item) => <Item key={item.key}>{item.label}</Item>}
        </ActionMenu>
    );
};

export const SourceReadOnly = ({ isActive, parameters, icon, menu, title }: SourceReadOnlyProps) => {
    return (
        <View padding={'size-250'} UNSAFE_className={isActive ? styles.sourceActive : styles.sourceInactive}>
            <Flex alignItems={'center'} gap={'size-200'}>
                {icon}
                <Heading margin={0}>{title}</Heading>
            </Flex>
            <Flex
                width={'100%'}
                justifyContent={'space-between'}
                marginTop={'size-200'}
                alignItems={'center'}
                minWidth={0}
                gap={'size-50'}
            >
                <ul className={styles.sourceReadOnlyList}>
                    {parameters.map((parameter) => (
                        <li key={parameter}>{parameter}</li>
                    ))}
                </ul>
                <View alignSelf={'end'}>{menu}</View>
            </Flex>
        </View>
    );
};

SourceReadOnly.Menu = SourceMenu;
