/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { ActionButton, Flex, Tooltip, TooltipTrigger } from '@geti/ui';
import { Close, Edit } from '@geti/ui/icons';
import { v4 as uuid } from 'uuid';

import { AddLabel } from './add-label.component';
import { LabelBadge } from './label-badge.component';
import { Label } from './label.interface';

import classes from './labels.module.css';

export const Labels = () => {
    const [labels, setLabels] = useState<Array<Label>>([
        { id: uuid(), name: 'Label 1', color: '#960F9F' },
        { id: uuid(), name: 'Label 2', color: '#0077BB' },
        { id: uuid(), name: 'Label 3', color: '#FDFAF0' },
    ]);

    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(null);

    const deleteLabels = (id: string) => setLabels(labels.filter((item) => item.id !== id));
    const editLabel = () => alert('Edit label - to be implemented');

    return (
        <Flex height={'100%'} alignItems={'center'} width={'100%'}>
            <Flex gap='size-200' margin={'size-50'} wrap={'wrap'} width={'100%'}>
                {labels.map((label) => (
                    <LabelBadge
                        onClick={() => setSelectedLabelId(label.id)}
                        key={label.id}
                        label={label}
                        isSelected={selectedLabelId === label.id}
                    >
                        <TooltipTrigger placement={'bottom'}>
                            <ActionButton
                                aria-label={`Edit ${label.name} label`}
                                isQuiet
                                UNSAFE_className={classes.iconButton}
                                onPress={editLabel}
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
                                onPress={() => deleteLabels(label.id)}
                            >
                                <Close />
                            </ActionButton>
                            <Tooltip>Delete label</Tooltip>
                        </TooltipTrigger>
                    </LabelBadge>
                ))}
                <Flex alignSelf={'flex-end'} flex={1} justifyContent={'end'}>
                    <AddLabel addLabel={(label) => setLabels([...labels, label])} />
                </Flex>
            </Flex>
        </Flex>
    );
};
