/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType } from '@/api';
import { ImagesFolder } from '@/icons';

import { PipelineEntityCard } from '../../pipeline-entity-card/pipeline-entity-card.component';

interface ImagesFolderSourceCardProps {
    source: ImagesFolderSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const ImagesFolderSourceCard = ({ source, onAction, menuItems }: ImagesFolderSourceCardProps) => {
    const parameters = [`Folder path: ${source.config.images_folder_path}`];
    const isActiveSource = source.active;

    return (
        <PipelineEntityCard
            isActive={isActiveSource}
            icon={<ImagesFolder />}
            title={'Images folder'}
            menu={<PipelineEntityCard.Menu onAction={onAction} isActive={isActiveSource} items={menuItems} />}
        >
            <PipelineEntityCard.Parameters parameters={parameters} />
        </PipelineEntityCard>
    );
};
