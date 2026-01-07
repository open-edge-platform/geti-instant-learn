/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Flex, Form } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';
import { isVideoFilePathValid } from './utils';
import { VideoFileFields } from './video-file-fields.component';

interface CreateVideoFileProps {
    onSaved: () => void;
}

export const CreateVideoFile = ({ onSaved }: CreateVideoFileProps) => {
    const createVideoFileSource = useCreateSource();
    const [filePath, setFilePath] = useState<string>('');
    const isSubmitDisabled = createVideoFileSource.isPending || !isVideoFilePathValid(filePath);

    const handleSubmit = (event: FormEvent) => {
        event.preventDefault();

        createVideoFileSource.mutate(
            {
                source_type: 'video_file',
                seekable: true,
                video_path: filePath.trim(),
            },
            onSaved
        );
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleSubmit}>
            <Flex direction={'column'} gap={'size-200'} marginTop={0}>
                <VideoFileFields filePath={filePath} onFilePathChange={setFilePath} />
                <ButtonGroup>
                    <Button type={'submit'} isPending={createVideoFileSource.isPending} isDisabled={isSubmitDisabled}>
                        Apply
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
