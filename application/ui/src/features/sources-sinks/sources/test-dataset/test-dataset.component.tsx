/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Flex, Heading, Text, View } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';

import styles from './test-dataset.module.scss';

interface TestDatasetProps {
    title: string;
    description: string;
    folderPath: string;
    imgSrc: string;
    onSaved: () => void;
}

export const TestDataset = ({ title, description, folderPath, imgSrc, onSaved }: TestDatasetProps) => {
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
                <img src={imgSrc} alt={title} className={styles.img} />
            </View>
            <View padding={'size-200'} backgroundColor={'gray-200'}>
                <Flex direction={'column'} gap={'size-200'}>
                    <Heading margin={0} UNSAFE_className={styles.title}>
                        {title}
                    </Heading>
                    <Text UNSAFE_className={styles.description}>{description}</Text>
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
