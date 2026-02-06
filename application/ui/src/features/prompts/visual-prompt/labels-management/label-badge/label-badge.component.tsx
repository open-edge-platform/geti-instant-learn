/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties, ReactNode } from 'react';

import { LabelType } from '@/api';
import { Text } from '@geti/ui';
import { clsx } from 'clsx';
import { usePress } from 'react-aria';

import classes from './label-badge.module.scss';

interface LabelIndicatorProps {
    label: LabelType;
    isSelected: boolean;
    onClick: () => void;
    children: ReactNode;
}

export const LabelBadge = ({ label, isSelected, onClick, children: actionButtons }: LabelIndicatorProps) => {
    const { pressProps } = usePress({
        onPress: onClick,
    });

    return (
        <div
            {...pressProps}
            style={{ '--labelBgColor': label.color } as CSSProperties}
            className={clsx(classes.badge, { [classes.selected]: isSelected })}
            aria-selected={isSelected}
            aria-label={`Label ${label.name}`}
        >
            <Text UNSAFE_className={classes.buttonText}>{label.name}</Text>
            {actionButtons}
        </div>
    );
};
