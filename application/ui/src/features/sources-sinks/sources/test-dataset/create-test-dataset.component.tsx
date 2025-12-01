/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Flex, Heading, Text, View } from '@geti/ui';

import TestDatasetImg from '../../../../assets/coffee-berries-placeholder.webp';
import { useCreateSource } from '../api/use-create-source';

import styles from './test-dataset.module.scss';

export const TestDatasetTitle = () => {
    return (
        <Heading margin={0} UNSAFE_className={styles.title}>
            Coffee Bean Quality Dataset
        </Heading>
    );
};

export const TestDatasetDescription = () => {
    return (
        <Text UNSAFE_className={styles.description}>
            This dataset contains information about the quality of coffee beans.
        </Text>
    );
};

interface CreateTestDatasetProps {
    folderPath: string;
    onSaved: () => void;
}

export const CreateTestDataset = ({ folderPath, onSaved }: CreateTestDatasetProps) => {
    const createTestDataset = useCreateSource();

    const handleCreateTestDataset = () => {
        createTestDataset.mutate(
            {
                images_folder_path: folderPath,
                seekable: true,
                source_type: 'images_folder',
            },
            onSaved
        );
    };

    const isApplyDisabled = createTestDataset.isPending;

    return (
        <View borderRadius={'small'}>
            <View>
                <img src={TestDatasetImg} alt={'Coffee Bean Quality Dataset'} className={styles.img} />
            </View>
            <View padding={'size-200'} backgroundColor={'gray-200'}>
                <Flex direction={'column'} gap={'size-200'}>
                    <TestDatasetTitle />
                    <TestDatasetDescription />
                    <Flex>
                        <Button
                            isPending={createTestDataset.isPending}
                            isDisabled={isApplyDisabled}
                            onPress={handleCreateTestDataset}
                        >
                            Apply
                        </Button>
                    </Flex>
                </Flex>
            </View>
        </View>
    );
};
