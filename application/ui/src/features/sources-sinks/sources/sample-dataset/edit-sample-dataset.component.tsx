/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { SampleDatasetSourceType } from '@/api';
import { Flex, Form, Item, Picker, Text, View } from '@geti/ui';
import { isNull } from 'lodash-es';

import { useUpdateSource } from '../api/use-update-source';
import { EditSourceButtons } from '../edit-sources/edit-source-buttons.component';
import { useAvailableDatasets } from './api/use-available-datasets';
import { SampleDatasetTitle } from './create-sample-dataset.component';

import styles from './sample-dataset.module.scss';

interface EditSampleDatasetContentProps {
    source: SampleDatasetSourceType;
    onSaved: () => void;
}

const EditSampleDatasetContent = ({ source, onSaved }: EditSampleDatasetContentProps) => {
    const [selectedDatasetId, setSelectedDatasetId] = useState<string>(source.config.dataset_id ?? '');
    const { data: datasets = [] } = useAvailableDatasets();
    const selectedDataset = datasets.find((dataset) => dataset.id === selectedDatasetId);
    const isActiveSource = source.active;

    const updateSampleDatasetSource = useUpdateSource();
    const isButtonDisabled =
        selectedDatasetId === source.config.dataset_id || !selectedDatasetId || updateSampleDatasetSource.isPending;

    const handleUpdateSampleDataset = (active: boolean) => {
        if (!selectedDatasetId) {
            return;
        }

        updateSampleDatasetSource.mutate(
            source.id,
            {
                config: {
                    source_type: 'sample_dataset',
                    seekable: true,
                    dataset_id: selectedDatasetId,
                },
                active,
            },
            onSaved
        );
    };

    const handleSave = () => {
        handleUpdateSampleDataset(source.active);
    };

    const handleSaveAndConnect = () => {
        handleUpdateSampleDataset(true);
    };

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        handleSave();
    };

    return (
        <View borderRadius={'small'}>
            <View>
                {selectedDataset?.thumbnail && (
                    <img src={selectedDataset.thumbnail} alt={selectedDataset.name} className={styles.img} />
                )}
            </View>
            <View padding={'size-200'} backgroundColor={'gray-200'}>
                <Form validationBehavior={'native'} onSubmit={handleSubmit}>
                    <Flex direction={'column'} gap={'size-200'}>
                        <SampleDatasetTitle text={selectedDataset?.name} />

                        {datasets.length > 0 ? (
                            <Picker
                                label={'Dataset'}
                                selectedKey={selectedDatasetId ?? undefined}
                                items={datasets}
                                onSelectionChange={(key) => !isNull(key) && setSelectedDatasetId(key as string)}
                            >
                                {(item) => <Item key={item.id}>{item.name}</Item>}
                            </Picker>
                        ) : (
                            <Text>No sample datasets are available.</Text>
                        )}

                        <EditSourceButtons
                            isActiveSource={isActiveSource}
                            onSave={handleSave}
                            onSaveAndConnect={handleSaveAndConnect}
                            isDisabled={isButtonDisabled}
                            isPending={updateSampleDatasetSource.isPending}
                        />
                    </Flex>
                </Form>
            </View>
        </View>
    );
};

interface EditSampleDatasetProps {
    source: SampleDatasetSourceType;
    onSaved: () => void;
}

export const EditSampleDataset = ({ source, onSaved }: EditSampleDatasetProps) => {
    return <EditSampleDatasetContent source={source} onSaved={onSaved} />;
};
