/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Button, Flex, Text } from '@geti/ui';
import { Close, Edit } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { useHover } from 'react-aria';

import { Label } from './label.interface';

import classes from './label-badge.module.css';

interface LabelIndicatorProps {
    label: Label;
    isSelected: boolean;
    onClick: () => void;
    deleteLabel: () => void;
}

export const LabelBadge = ({ label, isSelected, onClick, deleteLabel }: LabelIndicatorProps) => {
    //TODO: add tooltip to buttons

    const { hoverProps, isHovered } = useHover({});

    return (
        <div {...hoverProps} style={{ height: 'min-content' }}>
            <Button
                key={label.id}
                onPress={onClick}
                UNSAFE_style={{ backgroundColor: label.color, outlineColor: label.color }}
                UNSAFE_className={clsx(classes.badge, { [classes.selected]: isSelected })}
            >
                <Flex
                    UNSAFE_className={clsx(classes.buttonsContainer, { [classes.buttonsContainerOpened]: isHovered })}
                >
                    <ActionButton aria-label={`Edit ${label.name} label`} isQuiet UNSAFE_className={classes.iconButton}>
                        <Edit />
                    </ActionButton>
                    <ActionButton
                        aria-label={`Delete ${label.name} label`}
                        isQuiet
                        UNSAFE_className={classes.iconButton}
                        onPress={deleteLabel}
                    >
                        <Close />
                    </ActionButton>
                </Flex>

                <Text UNSAFE_className={classes.buttonText}>{label.name}</Text>
            </Button>
        </div>
    );
};
