/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Flex, Form } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';
import { VideoFileSourceFields } from './video-file-fields.component';

interface CreateVideoFileProps {
    onSaved: () => void;
}

export const CreateVideoFile = ({ onSaved }: CreateVideoFileProps) => {
    const createVideoFileSource = useCreateSource();
    const [filePath, setFilePath] = useState<string>('');
    const isPending = false;
    const isApplyDisabled = false;

    const handleApply = (event: FormEvent) => {
        event.preventDefault();

        createVideoFileSource.mutate(
            {
                source_type: 'video_file',
                seekable: true,
                video_path: filePath,
            },
            onSaved
        );
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleApply}>
            <Flex direction={'column'} gap={'size-200'} marginTop={0}>
                <VideoFileSourceFields filePath={filePath} onFilePathChange={setFilePath} />
                <ButtonGroup>
                    <Button type={'submit'} isPending={isPending} isDisabled={isApplyDisabled}>
                        Apply
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
