/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SampleDatasetSourceType } from '@geti-prompt/api';
import { Flex, View } from '@geti/ui';
import { Datasets } from '@geti/ui/icons';

import SampleDatasetImg from '../../../../assets/coffee-berries-placeholder.webp';
import { SourceCard } from '../source-card/source-card.component';
import { SampleDatasetDescription, SampleDatasetTitle } from './create-sample-dataset.component';

interface SampleDatasetCardProps {
    source: SampleDatasetSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const SampleDatasetCard = ({ source, onAction, menuItems }: SampleDatasetCardProps) => {
    const isActiveSource = source.connected;

    return (
        <SourceCard isActive={isActiveSource} icon={<Datasets width={'32px'} />} title={'Sample dataset'}>
            <Flex direction={'column'} gap={'size-200'}>
                <img src={SampleDatasetImg} alt={'Sample dataset'} style={{ display: 'block', width: '100%' }} />
                <SampleDatasetTitle />
                <Flex>
                    <SampleDatasetDescription />
                    <View alignSelf={'end'}>
                        <SourceCard.Menu isActive={isActiveSource} items={menuItems} onAction={onAction} />
                    </View>
                </Flex>
            </Flex>
        </SourceCard>
    );
};
