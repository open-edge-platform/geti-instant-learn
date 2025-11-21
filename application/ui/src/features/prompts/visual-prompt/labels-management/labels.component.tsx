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
    selectedLabelId: string | null;
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
        <Flex direction={'column'} gap={'size-100'} width={'100%'}>
            <AddLabel existingLabels={labels} />
            <Flex direction={'column'} gap={'size-50'} width={'100%'}>
                <LabelsList labels={labels} selectedLabelId={selectedLabelId} setSelectedLabelId={setSelectedLabelId} />
            </Flex>
        </Flex>
    );
};
