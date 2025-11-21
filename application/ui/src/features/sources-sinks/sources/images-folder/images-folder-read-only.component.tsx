/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ImagesFolderSourceType } from '@geti-prompt/api';
import { ImagesFolder } from '@geti-prompt/icons';

import { SourceReadOnly } from '../source-read-only/source-read-only.component';

interface ImagesFolderSourceReadOnlyProps {
    isActive: boolean;
    source: ImagesFolderSourceType;
    menuItems: { key: string; label: string }[];
    onAction: (action: string) => void;
}

export const ImagesFolderSourceReadOnly = ({
    isActive,
    source,
    onAction,
    menuItems,
}: ImagesFolderSourceReadOnlyProps) => {
    const parameters = [`Folder path: ${source.config.images_folder_path}`];

    return (
        <SourceReadOnly
            isActive={isActive}
            parameters={parameters}
            icon={<ImagesFolder />}
            title={'Images folder'}
            menu={<SourceReadOnly.Menu onAction={onAction} isActive={isActive} items={menuItems} />}
        />
    );
};
