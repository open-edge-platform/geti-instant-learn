/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { Flex } from '@geti/ui';

import { AddLabel } from './add-label.component';
import { Label } from './label.interface';

export const Labels = () => {
    const [labels, setLabels] = useState<Array<Label>>([
        { id: '1', name: 'Label 1', color: '#ff0000' },
        { id: '2', name: 'Label 2', color: '#00ff00' },
        { id: '3', name: 'Label 3', color: '#0000ff' },
    ]);

    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'space-between'}>
            <Flex>
                {labels.map((label) => (
                    <div
                        key={label.id}
                        style={{ backgroundColor: label.color, padding: '4px', margin: '2px', borderRadius: '4px' }}
                    >
                        {label.name}
                    </div>
                ))}
            </Flex>
            <AddLabel addLabel={(label) => setLabels([...labels, label])} />
        </Flex>
    );
};
