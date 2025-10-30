/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { LabelType } from '@geti-prompt/api';
import { useProjectLabels } from '@geti-prompt/hooks';
import { Flex } from '@geti/ui';
import { useAnnotator } from 'src/features/annotator/providers/annotator-provider.component';

import { AddLabel } from './add-label/add-label.component';
import { LabelListItem } from './label-list-item/label-list-item.component';

interface LabelsListProps {
    labels: LabelType[];
    selectedLabel: LabelType;
    setSelectedLabel: (label: LabelType) => void;
}

const LabelsList = ({ labels, selectedLabel, setSelectedLabel }: LabelsListProps) => {
    return labels.map((label) => (
        <LabelListItem
            key={label.id}
            label={label}
            onSelect={() => setSelectedLabel(label)}
            isSelected={selectedLabel?.id === label.id}
            existingLabels={labels}
        />
    ));
};

export const Labels = () => {
    const { selectedLabel, setSelectedLabel } = useAnnotator();
    const labels = useProjectLabels();

    return (
        <Flex height={'100%'} alignItems={'center'} width={'100%'}>
            <Flex margin={'size-50'} wrap={'wrap'} width={'100%'} alignItems={'center'} gap={'size-100'}>
                <LabelsList labels={labels} selectedLabel={selectedLabel} setSelectedLabel={setSelectedLabel} />
                <Flex alignSelf={'flex-end'} flex={1} justifyContent={'end'} alignItems={'center'}>
                    <AddLabel existingLabels={labels} />
                </Flex>
            </Flex>
        </Flex>
    );
};
