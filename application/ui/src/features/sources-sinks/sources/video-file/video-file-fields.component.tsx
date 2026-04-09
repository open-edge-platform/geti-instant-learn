/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Flex, TextField, View } from '@geti/ui';

import { isTauriRuntime, pickVideoFilePath } from '../../../../shared/tauri/file-picker';

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

    const isTauri = isTauriRuntime();

    return (
        <View>
            <Flex alignItems={'end'} gap={'size-100'}>
                <TextField label={'File path'} isRequired value={filePath} onChange={onFilePathChange} width={'100%'} />
                <Button variant={'secondary'} onPress={handleBrowse} isDisabled={!isTauri}>
                    Browse
                </Button>
            </Flex>
        </View>
    );
};
