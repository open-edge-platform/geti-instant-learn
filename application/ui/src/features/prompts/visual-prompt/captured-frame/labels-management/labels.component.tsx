/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api, LabelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Flex } from '@geti/ui';

import { AddLabel } from './add-label/add-label.component';
import { LabelListItem } from './label-list-item/label-list-item.component';

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

interface LabelsListProps {
    labels: LabelType[];
}

const LabelsList = ({ labels }: LabelsListProps) => {
    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(null);

    return (
        <>
            {labels.map((label) => (
                <LabelListItem
                    key={label.id}
                    label={label}
                    onSelect={() => setSelectedLabelId(label.id)}
                    isSelected={selectedLabelId === label.id}
                    existingLabelsNames={labels.filter(({ id }) => id !== label.id).map(({ name }) => name)}
                />
            ))}
        </>
    );
};

export const Labels = () => {
    const { data } = useLabelsQuery();

    const existingLabelsNames = data.labels.map((label) => label.name);

    return (
        <Flex height={'100%'} alignItems={'center'} width={'100%'}>
            <Flex margin={'size-50'} wrap={'wrap'} width={'100%'} alignItems={'center'} gap={'size-100'}>
                <LabelsList labels={data.labels} />
                <Flex alignSelf={'flex-end'} flex={1} justifyContent={'end'} alignItems={'center'}>
                    <AddLabel existingLabelsNames={existingLabelsNames} />
                </Flex>
            </Flex>
        </Flex>
    );
};
