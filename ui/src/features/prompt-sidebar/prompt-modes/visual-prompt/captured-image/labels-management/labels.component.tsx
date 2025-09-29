/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { Flex, View } from '@geti/ui';
import { v4 as uuid } from 'uuid';

import { AddLabel } from './add-label.component';
import { LabelBadge } from './label-badge.component';
import { Label } from './label.interface';
import { getDistinctColorBasedOnHash } from './utils-temp';

export const Labels = () => {
    const [labels, setLabels] = useState<Array<Label>>([
        { id: uuid(), name: 'Label 1', color: getDistinctColorBasedOnHash('Label 1') },
        { id: uuid(), name: 'Label 2', color: getDistinctColorBasedOnHash('Label 2') },
    ]);

    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(null);

    return (
        <Flex height={'100%'} alignItems={'center'} width={'100%'}>
            <Flex gap='size-200' margin={'size-50'} wrap={'wrap'} width={'100%'}>
                {labels.map((label) => (
                    <LabelBadge
                        onClick={() => setSelectedLabelId(label.id)}
                        key={label.id}
                        label={label}
                        isSelected={selectedLabelId === label.id}
                        deleteLabel={() => setLabels(labels.filter((item) => item.id !== label.id))}
                    />
                ))}
                <Flex alignSelf={'flex-end'} flex={1} justifyContent={'end'}>
                    <AddLabel addLabel={(label) => setLabels([...labels, label])} />
                </Flex>
            </Flex>
        </Flex>
    );
};
