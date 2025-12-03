/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { ActionMenu, Flex, Heading, Item, View } from '@geti/ui';
import { clsx } from 'clsx';

import styles from './pipeline-entity-card.module.scss';

interface PipelineEntityMenuProps {
    isActive: boolean;
    onAction: (action: string) => void;
    items: { key: string; label: string }[];
}

const PipelineEntityMenu = ({ onAction, isActive, items }: PipelineEntityMenuProps) => {
    return (
        <ActionMenu
            isQuiet
            UNSAFE_className={clsx(styles.menu, {
                [styles.activeMenu]: isActive,
            })}
            onAction={(key) => onAction(String(key))}
            items={items}
        >
            {(item) => <Item key={item.key}>{item.label}</Item>}
        </ActionMenu>
    );
};

interface PipelineEntityCardProps {
    isActive: boolean;
    children: ReactNode;
    icon: ReactNode;
    title: string;
    menu?: ReactNode;
}

const PipelineEntityCardParametersList = ({ parameters }: { parameters: string[] }) => {
    return (
        <ul className={styles.parametersList}>
            {parameters.map((parameter) => (
                <li key={parameter}>{parameter}</li>
            ))}
        </ul>
    );
};

export const PipelineEntityCard = ({ isActive, children, icon, menu, title }: PipelineEntityCardProps) => {
    return (
        <View padding={'size-250'} UNSAFE_className={isActive ? styles.active : styles.inactive}>
            <Flex alignItems={'center'} gap={'size-100'}>
                {icon}
                <Heading margin={0} UNSAFE_className={styles.title}>
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

PipelineEntityCard.Menu = PipelineEntityMenu;
PipelineEntityCard.Parameters = PipelineEntityCardParametersList;
