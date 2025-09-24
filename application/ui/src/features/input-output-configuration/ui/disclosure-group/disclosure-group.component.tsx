/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, useState } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex, Text } from '@geti/ui';
import { clsx } from 'clsx';

import styles from './disclosure-group.module.scss';

interface DisclosureGroupProps<Value extends string> {
    value: Value | null;
    onChange?: (value: Value) => void;
    items: { value: Value; label: string; icon: ReactNode; content?: ReactNode }[];
}

interface DisclosureItemProps<Value extends string> {
    item: { value: Value; label: string; icon: ReactNode; content?: ReactNode };
    onChange?: (value: Value) => void;
    value: Value | null;
}

const DisclosureItem = <Value extends string>({ item, value, onChange }: DisclosureItemProps<Value>) => {
    const [isExpanded, setIsExpanded] = useState<boolean>(item.value === value);

    const handleExpandedChange = (expanded: boolean) => {
        setIsExpanded(expanded);
        onChange !== undefined && onChange(item.value);
    };

    return (
        <Disclosure
            isQuiet
            key={item.label}
            isExpanded={isExpanded}
            UNSAFE_className={clsx(styles.disclosure, {
                [styles.selected]: item.value === value,
            })}
            onExpandedChange={handleExpandedChange}
        >
            <DisclosureTitle UNSAFE_className={styles.disclosureTitleContainer}>
                <Flex alignItems={'center'} justifyContent={'space-between'} width={'100%'}>
                    <Flex marginStart={'size-50'} alignItems={'center'} gap={'size-100'}>
                        {item.icon}
                        <Text UNSAFE_className={styles.disclosureTitle}>{item.label}</Text>
                    </Flex>
                </Flex>
            </DisclosureTitle>
            <DisclosurePanel>{isExpanded && item.content}</DisclosurePanel>
        </Disclosure>
    );
};

export const DisclosureGroup = <Value extends string>({ onChange, items, value }: DisclosureGroupProps<Value>) => {
    return (
        <Flex width={'100%'} direction={'column'} gap={'size-100'}>
            {items.map((item) => (
                <DisclosureItem item={item} key={item.label} onChange={onChange} value={value} />
            ))}
        </Flex>
    );
};
