/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType } from '@geti-prompt/api';
import { Flex, View } from '@geti/ui';
import { Datasets } from '@geti/ui/icons';

import TestDatasetImg from '../../../../assets/coffee-berries-placeholder.webp';
import { SourceCard } from '../source-card/source-card.component';
import { TestDatasetDescription, TestDatasetTitle } from './create-test-dataset.component';

interface TestDatasetCardProps {
    source: ImagesFolderSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const TestDatasetCard = ({ source, onAction, menuItems }: TestDatasetCardProps) => {
    const isActiveSource = source.connected;

    return (
        <SourceCard isActive={isActiveSource} icon={<Datasets width={'32px'} />} title={'Test dataset'}>
            <Flex direction={'column'} gap={'size-200'}>
                <img src={TestDatasetImg} alt={'Test dataset'} style={{ display: 'block', width: '100%' }} />
                <TestDatasetTitle />
                <Flex>
                    <TestDatasetDescription />
                    <View alignSelf={'end'}>
                        <SourceCard.Menu isActive={isActiveSource} items={menuItems} onAction={onAction} />
                    </View>
                </Flex>
            </Flex>
        </SourceCard>
    );
};
