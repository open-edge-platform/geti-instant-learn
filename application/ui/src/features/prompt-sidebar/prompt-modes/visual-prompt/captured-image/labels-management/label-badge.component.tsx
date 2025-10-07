/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Flex, Text, Tooltip, TooltipTrigger } from '@geti/ui';
import { Close, Edit } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { useHover, usePress } from 'react-aria';

import { Label } from './label.interface';

import classes from './label-badge.module.css';

interface LabelIndicatorProps {
    label: Label;
    isSelected: boolean;
    onClick: () => void;
    deleteLabel: () => void;
}

export const LabelBadge = ({ label, isSelected, onClick, deleteLabel }: LabelIndicatorProps) => {
    const { hoverProps, isHovered } = useHover({});
    const { pressProps } = usePress({
        onPress: onClick,
    });

    return (
        <div {...hoverProps} style={{ height: 'min-content' }}>
            <div
                key={label.id}
                {...pressProps}
                style={{ '--labelBgColor': label.color } as React.CSSProperties}
                className={clsx(classes.badge, { [classes.selected]: isSelected })}
            >
                <Text UNSAFE_className={classes.buttonText}>{label.name}</Text>
                <Flex
                    UNSAFE_className={clsx(classes.buttonsContainer, { [classes.buttonsContainerOpened]: isHovered })}
                >
                    <TooltipTrigger placement={'bottom'}>
                        <ActionButton
                            aria-label={`Edit ${label.name} label`}
                            isQuiet
                            UNSAFE_className={classes.iconButton}
                        >
                            <Edit />
                        </ActionButton>
                        <Tooltip>Edit label name</Tooltip>
                    </TooltipTrigger>
                    <TooltipTrigger placement={'bottom'}>
                        <ActionButton
                            aria-label={`Delete ${label.name} label`}
                            isQuiet
                            UNSAFE_className={classes.iconButton}
                            onPress={deleteLabel}
                        >
                            <Close />
                        </ActionButton>
                        <Tooltip>Delete label</Tooltip>
                    </TooltipTrigger>
                </Flex>
            </div>
        </div>
    );
};
