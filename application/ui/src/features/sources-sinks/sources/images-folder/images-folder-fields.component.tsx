/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, ContextualHelp, Heading, Text, TextField } from '@geti/ui';

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
    return (
        <TextField
            label={'Folder path'}
            value={folderPath}
            onChange={onSetFolderPath}
            width={'100%'}
            contextualHelp={<FolderPathDescription />}
        />
    );
};
