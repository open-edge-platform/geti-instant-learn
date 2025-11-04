/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { ImagesFolderConfig } from '@geti-prompt/api';
import { Folder } from '@geti-prompt/icons';
import { ActionButton, Button, Flex, TextField, View } from '@geti/ui';

import styles from './images-folder.module.scss';

interface ImagesFolderProps {
    source: ImagesFolderConfig | undefined;
}

const validatePath = (path: string) => {
    if (path.trim().length === 0) {
        return 'Folder path cannot be empty';
    }

    return undefined;
};

export const ImagesFolder = ({ source }: ImagesFolderProps) => {
    const [folderPath, setFolderPath] = useState(source?.config.images_folder_path ?? '');

    const isApplyDisabled = folderPath.trim().length === 0;

    return (
        <View>
            <Flex alignItems={'end'} gap={'size-50'}>
                <TextField
                    label={'Folder path'}
                    value={folderPath}
                    onChange={setFolderPath}
                    //validate={validatePath}
                    flex={1}
                />
                <ActionButton isQuiet aria-label={'Select folder'} UNSAFE_className={styles.imagesFolderButton}>
                    <Folder />
                </ActionButton>
            </Flex>

            <Button variant={'accent'} isDisabled={isApplyDisabled} marginTop={'size-200'}>
                Apply
            </Button>
        </View>
    );
};
