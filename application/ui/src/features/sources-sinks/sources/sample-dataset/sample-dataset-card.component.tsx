/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SampleDatasetSourceType } from '@/api';
import { Flex, View } from '@geti/ui';
import { Datasets } from '@geti/ui/icons';

import { PipelineEntityCard } from '../../pipeline-entity-card/pipeline-entity-card.component';
import { useAvailableDatasets } from './api/use-available-datasets';
import { SampleDatasetTitle } from './create-sample-dataset.component';

interface SampleDatasetCardProps {
    source: SampleDatasetSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const SampleDatasetCard = ({ source, onAction, menuItems }: SampleDatasetCardProps) => {
    const isActiveSource = source.active;
    const { data: datasets = [] } = useAvailableDatasets();
    const selectedDataset = datasets.find((dataset) => dataset.id === source.config.dataset_id);

    if (selectedDataset === undefined) {
        return null;
    }

    const thumbnail = selectedDataset.thumbnail ?? undefined;

    return (
        <PipelineEntityCard isActive={isActiveSource} icon={<Datasets width={'32px'} />} title={'Sample dataset'}>
            <Flex direction={'column'} gap={'size-200'}>
                {thumbnail && (
                    <img src={thumbnail} alt={selectedDataset.name} style={{ display: 'block', width: '100%' }} />
                )}
                <Flex justifyContent={'space-between'} alignItems={'end'} gap={'size-100'}>
                    <SampleDatasetTitle text={selectedDataset.name} />
                    <View alignSelf={'end'}>
                        <PipelineEntityCard.Menu isActive={isActiveSource} items={menuItems} onAction={onAction} />
                    </View>
                </Flex>
            </Flex>
        </PipelineEntityCard>
    );
};
