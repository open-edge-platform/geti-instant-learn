/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType } from '@geti-prompt/api';
import { Datasets } from '@geti/ui/icons';

import { SourceCard } from '../source-card/source-card.component';

interface TestDatasetCardProps {
    source: ImagesFolderSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const TestDatasetCard = ({ source, onAction, menuItems }: TestDatasetCardProps) => {
    const parameters = [`Folder path: ${source.config.images_folder_path}`];
    const isActiveSource = source.connected;

    return (
        <SourceCard
            isActive={isActiveSource}
            parameters={<SourceCard.Parameters parameters={parameters} />}
            icon={<Datasets width={'32px'} />}
            title={'Test dataset'}
            menu={<SourceCard.Menu isActive={isActiveSource} items={menuItems} onAction={onAction} />}
        />
    );
};
