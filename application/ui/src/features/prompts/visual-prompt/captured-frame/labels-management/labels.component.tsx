/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Flex } from '@geti/ui';

import { AddLabel } from './add-label.component';
import { LabelListItem } from './label-list-item.component';

const useLabelsQuery = () => {
    const { projectId } = useProjectIdentifier();
    return $api.useSuspenseQuery('get', '/api/v1/projects/{project_id}/labels', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });
};

const LabelsList = () => {
    const { data } = useLabelsQuery();

    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(null);

    return (
        <>
            {data.labels.map((label) => (
                <LabelListItem
                    key={label.id}
                    label={label}
                    onSelect={() => setSelectedLabelId(label.id)}
                    isSelected={selectedLabelId === label.id}
                />
            ))}
        </>
    );
};

export const Labels = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} width={'100%'}>
            <Flex margin={'size-50'} wrap={'wrap'} width={'100%'} alignItems={'center'} gap={'size-100'}>
                <LabelsList />
                <Flex alignSelf={'flex-end'} flex={1} justifyContent={'end'} alignItems={'center'}>
                    <AddLabel />
                </Flex>
            </Flex>
        </Flex>
    );
};
