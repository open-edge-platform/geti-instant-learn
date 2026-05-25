/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, TextField, View } from '@geti/ui';

interface VideoFileFieldsProps {
    filePath: string;
    onFilePathChange: (value: string) => void;
}

export const VideoFileFields = ({ filePath, onFilePathChange }: VideoFileFieldsProps) => {
    return (
        <View>
            <Flex alignItems={'end'} gap={'size-100'}>
                <TextField label={'File path'} isRequired value={filePath} onChange={onFilePathChange} width={'100%'} />
            </Flex>
        </View>
    );
};
