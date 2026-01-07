/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { TextField, View } from '@geti/ui';

interface VideoFileFieldsProps {
    filePath: string;
    onFilePathChange: (value: string) => void;
}

export const VideoFileFields = ({ filePath, onFilePathChange }: VideoFileFieldsProps) => {
    return (
        <View>
            <TextField label={'File path'} isRequired value={filePath} onChange={onFilePathChange} width={'100%'} />
        </View>
    );
};
