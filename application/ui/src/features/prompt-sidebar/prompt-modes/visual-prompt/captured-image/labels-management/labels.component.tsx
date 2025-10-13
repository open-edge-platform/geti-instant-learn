/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { Flex } from '@geti/ui';
import { v4 as uuid } from 'uuid';

import { AddLabel } from './add-label.component';
import { LabelListItem } from './label-list-item.component';
import { Label } from './label.interface';

export const Labels = () => {
    const [labels, setLabels] = useState<Array<Label>>([
        { id: uuid(), name: 'Label 1', color: '#960F9F' },
        { id: uuid(), name: 'Label 2', color: '#0077BB' },
        { id: uuid(), name: 'Label 3', color: '#FDFAF0' },
    ]);

    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(null);

    const deleteLabel = (id: string) => setLabels(labels.filter((item) => item.id !== id));

    return (
        <Flex height={'100%'} alignItems={'center'} width={'100%'}>
            <Flex margin={'size-50'} wrap={'wrap'} width={'100%'} alignItems={'center'}>
                {labels.map((label) => (
                    <LabelListItem
                        key={label.id}
                        label={label}
                        deleteLabel={() => deleteLabel(label.id)}
                        onSelect={() => setSelectedLabelId(label.id)}
                        isSelected={selectedLabelId === label.id}
                        onUpdate={(edited: Label) =>
                            setLabels(labels.map((item) => (item.id === label.id ? edited : item)))
                        }
                    />
                ))}
                <Flex alignSelf={'flex-end'} flex={1} justifyContent={'end'} alignItems={'center'}>
                    <AddLabel onAddLabel={(label) => setLabels([...labels, label])} />
                </Flex>
            </Flex>
        </Flex>
    );
};
