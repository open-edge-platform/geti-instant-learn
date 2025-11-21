/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType } from '@geti-prompt/api';
import { ImagesFolder } from '@geti-prompt/icons';

import { SourceCard } from '../source-card/source-card.component';

interface ImagesFolderSourceCardProps {
    isActive: boolean;
    source: ImagesFolderSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const ImagesFolderSourceCard = ({ isActive, source, onAction, menuItems }: ImagesFolderSourceCardProps) => {
    const parameters = [`Folder path: ${source.config.images_folder_path}`];

    return (
        <SourceCard
            isActive={isActive}
            parameters={parameters}
            icon={<ImagesFolder />}
            title={'Images folder'}
            menu={<SourceCard.Menu onAction={onAction} isActive={isActive} items={menuItems} />}
        />
    );
};
