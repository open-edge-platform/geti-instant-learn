/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Flex, TextField, View } from '@geti/ui';
import { open } from '@tauri-apps/plugin-dialog';

const getSingleSelectedPath = (selectedPath: string | string[] | null): string | null => {
    if (typeof selectedPath === 'string') {
        return selectedPath;
    }

    if (Array.isArray(selectedPath)) {
        return selectedPath.length > 0 ? selectedPath[0] : null;
    }
    return null;
};

export const pickVideoFilePath = async (): Promise<string | null> => {
    const selectedPath = await open({
        directory: false,
        multiple: false,
        filters: [
            {
                name: 'Videos',
                extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v'],
            },
        ],
    });

    return getSingleSelectedPath(selectedPath);
};

interface VideoFileFieldsProps {
    filePath: string;
    onFilePathChange: (value: string) => void;
}

export const VideoFileFields = ({ filePath, onFilePathChange }: VideoFileFieldsProps) => {
    const handleBrowse = async (): Promise<void> => {
        const selectedPath = await pickVideoFilePath();

        if (selectedPath !== null) {
            onFilePathChange(selectedPath);
        }
    };

    return (
        <View>
            <Flex alignItems={'end'} gap={'size-100'}>
                <TextField label={'File path'} isRequired value={filePath} onChange={onFilePathChange} width={'100%'} />
                <Button variant={'secondary'} onPress={handleBrowse}>
                    Browse
                </Button>
            </Flex>
        </View>
    );
};
