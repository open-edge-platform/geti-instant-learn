/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Flex, Heading, Text, View } from '@geti/ui';

import TestDatasetImg from '../../../../assets/coffee-berries-placeholder.webp';
import { useCreateSource } from '../api/use-create-source';

import styles from './sample-dataset.module.scss';

export const SampleDatasetTitle = () => {
    return (
        <Heading margin={0} UNSAFE_className={styles.title}>
            Coffee Bean Quality Dataset
        </Heading>
    );
};

export const SampleDatasetDescription = () => {
    return (
        <Text UNSAFE_className={styles.description}>
            This dataset contains information about the quality of coffee beans.
        </Text>
    );
};

interface CreateSampleDatasetProps {
    onSaved: () => void;
}

export const CreateSampleDataset = ({ onSaved }: CreateSampleDatasetProps) => {
    const createSampleDataset = useCreateSource();

    const handleCreateSampleDataset = () => {
        createSampleDataset.mutate(
            {
                seekable: true,
                source_type: 'sample_dataset',
            },
            onSaved
        );
    };

    const isApplyDisabled = createSampleDataset.isPending;

    return (
        <View borderRadius={'small'}>
            <View>
                <img src={TestDatasetImg} alt={'Coffee Bean Quality Dataset'} className={styles.img} />
            </View>
            <View padding={'size-200'} backgroundColor={'gray-200'}>
                <Flex direction={'column'} gap={'size-200'}>
                    <SampleDatasetTitle />
                    <SampleDatasetDescription />
                    <Flex>
                        <Button
                            isPending={createSampleDataset.isPending}
                            isDisabled={isApplyDisabled}
                            onPress={handleCreateSampleDataset}
                        >
                            Apply
                        </Button>
                    </Flex>
                </Flex>
            </View>
        </View>
    );
};
