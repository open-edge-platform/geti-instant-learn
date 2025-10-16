/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { ActionButton, Tooltip, TooltipTrigger } from '@geti/ui';
import { Close, Edit } from '@geti/ui/icons';

import { EditLabel } from './edit-label.component';
import { LabelBadge } from './label-badge.component';
import { Label } from './label.interface';

import classes from './label-list-item.module.css';

interface LabelListItemProps {
    label: Label;
    deleteLabel: () => void;
    onSelect: () => void;
    isSelected: boolean;
    onUpdate: (edited: Label) => void;
}

export const LabelListItem = ({ label, deleteLabel, onSelect, isSelected, onUpdate }: LabelListItemProps) => {
    const [isInEdition, setIsInEdition] = useState<boolean>(false);

    if (!isInEdition) {
        return (
            <LabelBadge onClick={onSelect} key={label.id} label={label} isSelected={isSelected}>
                <TooltipTrigger placement={'bottom'}>
                    <ActionButton
                        aria-label={`Edit ${label.name} label`}
                        isQuiet
                        UNSAFE_className={classes.iconButton}
                        onPress={() => setIsInEdition(true)}
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
            </LabelBadge>
        );
    } else {
        return (
            <EditLabel
                onAccept={onUpdate}
                onClose={() => setIsInEdition(false)}
                label={label}
                isQuiet
                width={'size-2400'}
            />
        );
    }
};
