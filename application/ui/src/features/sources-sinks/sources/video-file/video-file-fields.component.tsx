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
    const isTauri = isTauriRuntime();

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
                {isTauri ? (
                    <Button variant={'secondary'} onPress={handleBrowse}>
                        Browse
                    </Button>
                ) : null}
            </Flex>
        </View>
    );
};
