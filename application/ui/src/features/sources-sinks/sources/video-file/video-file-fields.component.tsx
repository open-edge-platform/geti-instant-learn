/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { TextField, View } from '@geti/ui';

interface VideoFileSourceFieldsProps {
    filePath: string;
    onFilePathChange: (value: string) => void;
}

export const VideoFileSourceFields = ({ filePath, onFilePathChange }: VideoFileSourceFieldsProps) => {
    return (
        <View>
            <TextField label={'File path'} isRequired value={filePath} onChange={onFilePathChange} />
        </View>
    );
};
