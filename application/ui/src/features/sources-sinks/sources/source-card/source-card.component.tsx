/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { ActionMenu, Flex, Heading, Item, View } from '@geti/ui';
import { clsx } from 'clsx';

import styles from './source-card.module.scss';

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

interface SourceCardProps {
    isActive: boolean;
    children: ReactNode;
    icon: ReactNode;
    title: string;
    menu?: ReactNode;
}

const SourceCardParametersList = ({ parameters }: { parameters: string[] }) => {
    return (
        <ul className={styles.sourceParametersList}>
            {parameters.map((parameter) => (
                <li key={parameter}>{parameter}</li>
            ))}
        </ul>
    );
};

export const SourceCard = ({ isActive, children, icon, menu, title }: SourceCardProps) => {
    return (
        <View padding={'size-250'} UNSAFE_className={isActive ? styles.sourceActive : styles.sourceInactive}>
            <Flex alignItems={'center'} gap={'size-100'}>
                {icon}
                <Heading margin={0} UNSAFE_className={styles.sourceTitle}>
                    {title}
                </Heading>
            </Flex>
            <Flex
                width={'100%'}
                justifyContent={'space-between'}
                marginTop={'size-200'}
                alignItems={'center'}
                minWidth={0}
                gap={'size-50'}
            >
                <View flex={1}>{children}</View>
                <View alignSelf={'end'}>{menu}</View>
            </Flex>
        </View>
    );
};

SourceCard.Menu = SourceMenu;
SourceCard.Parameters = SourceCardParametersList;
