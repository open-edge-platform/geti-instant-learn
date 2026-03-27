/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Content, ContextualHelp, Flex, Heading, Text, TextField } from '@geti/ui';

import { isTauriRuntime, pickFolderPath } from '../../../../shared/tauri/file-picker';

const FolderPathDescription = () => {
    return (
        <ContextualHelp variant='info'>
            <Heading>What is a folder path?</Heading>
            <Content>
                <Text>
                    A folder path is the location of a directory on your system.
                    <br />
                    Enter the absolute path (e.g. /Users/username/images) or relative path (e.g. ./data/images) to the
                    folder containing your images.
                </Text>
            </Content>
        </ContextualHelp>
    );
};

interface ImagesFolderFieldsProps {
    folderPath: string;
    onSetFolderPath: (path: string) => void;
}

export const ImagesFolderFields = ({ folderPath, onSetFolderPath }: ImagesFolderFieldsProps) => {
    const handleBrowse = async (): Promise<void> => {
        const selectedPath = await pickFolderPath();

        if (selectedPath !== null) {
            onSetFolderPath(selectedPath);
        }
    };

    const isTauri = isTauriRuntime();

    return (
        <Flex alignItems={'end'} gap={'size-100'}>
            <TextField
                label={'Folder path'}
                value={folderPath}
                onChange={onSetFolderPath}
                width={'100%'}
                contextualHelp={<FolderPathDescription />}
                isRequired
            />
            <Button variant={'secondary'} onPress={handleBrowse} isDisabled={!isTauri}>
                Browse
            </Button>
        </Flex>
    );
};
