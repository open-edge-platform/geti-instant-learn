/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType } from '@geti-prompt/api';
import { Flex } from '@geti/ui';

import { useVisualPrompt } from '../visual-prompt-provider.component';
import { AddLabel } from './add-label/add-label.component';
import { LabelListItem } from './label-list-item/label-list-item.component';

interface LabelsListProps {
    labels: LabelType[];
    selectedLabelId: string;
    setSelectedLabelId: (label: string) => void;
}

const LabelsList = ({ labels, selectedLabelId, setSelectedLabelId }: LabelsListProps) => {
    return labels.map((label) => (
        <LabelListItem
            key={label.id}
            label={label}
            onSelect={() => setSelectedLabelId(label.id)}
            isSelected={selectedLabelId === label.id}
            existingLabels={labels}
        />
    ));
};

export const Labels = () => {
    const { selectedLabelId, setSelectedLabelId, labels } = useVisualPrompt();

    return (
        <Flex height={'100%'} alignItems={'center'} width={'100%'}>
            <Flex margin={'size-50'} wrap={'wrap'} width={'100%'} alignItems={'center'} gap={'size-100'}>
                <LabelsList labels={labels} selectedLabelId={selectedLabelId} setSelectedLabelId={setSelectedLabelId} />
                <Flex alignSelf={'flex-end'} flex={1} justifyContent={'end'} alignItems={'center'}>
                    <AddLabel existingLabels={labels} />
                </Flex>
            </Flex>
        </Flex>
    );
};
