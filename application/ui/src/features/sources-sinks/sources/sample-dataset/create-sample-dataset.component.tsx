/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from 'react';

import { Button, ButtonGroup, Flex, Form, Heading, Item, Picker, Text, View } from '@geti/ui';
import { isNull } from 'lodash-es';

import { useCreateSource } from '../api/use-create-source';
import { useAvailableDatasets } from './api/use-available-datasets';

import styles from './sample-dataset.module.scss';

interface SampleDatasetTextProps {
    text?: string;
}

export const SampleDatasetTitle = ({ text = 'Sample dataset' }: SampleDatasetTextProps) => {
    return (
        <Heading margin={0} UNSAFE_className={styles.title}>
            {text}
        </Heading>
    );
};

export const SampleDatasetDescription = ({
    text = 'A collection of images for visual AI training and inference.',
}: SampleDatasetTextProps) => {
    return <Text UNSAFE_className={styles.description}>{text}</Text>;
};

interface CreateSampleDatasetProps {
    onSaved: () => void;
}

export const CreateSampleDataset = ({ onSaved }: CreateSampleDatasetProps) => {
    const createSampleDataset = useCreateSource();
    const { data: datasets = [], isPending: isLoadingDatasets } = useAvailableDatasets();
    const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
    const selectedDataset = datasets.find((dataset) => dataset.id === selectedDatasetId);

    useEffect(() => {
        if (selectedDatasetId === null && datasets.length > 0) {
            setSelectedDatasetId(datasets[0].id);
        }
    }, [datasets, selectedDatasetId]);

    const handleCreateSampleDataset = () => {
        if (!selectedDatasetId) {
            return;
        }

        createSampleDataset.mutate(
            {
                seekable: true,
                source_type: 'sample_dataset',
                dataset_id: selectedDatasetId,
            },
            onSaved
        );
    };

    const isApplyDisabled = createSampleDataset.isPending || isLoadingDatasets || !selectedDatasetId;

    return (
        <View borderRadius={'small'}>
            <View>
                {selectedDataset?.thumbnail && (
                    <img
                        src={selectedDataset.thumbnail}
                        alt={selectedDataset.name}
                        className={styles.img}
                    />
                )}
            </View>
            <View padding={'size-200'} backgroundColor={'gray-200'}>
                <Form validationBehavior={'native'} onSubmit={handleCreateSampleDataset}>
                    <Flex direction={'column'} gap={'size-200'}>
                        <SampleDatasetTitle text={selectedDataset?.name} />
                        <SampleDatasetDescription text={selectedDataset?.description} />

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

                        <ButtonGroup>
                            <Button
                                type={'submit'}
                                isPending={createSampleDataset.isPending}
                                isDisabled={isApplyDisabled}
                            >
                                Apply
                            </Button>
                        </ButtonGroup>
                    </Flex>
                </Form>
            </View>
        </View>
    );
};
