/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { VideoFileSourceType } from '@/api';
import { Flex, Form } from '@geti/ui';

import { useUpdateSource } from '../api/use-update-source';
import { EditSourceButtons } from '../edit-sources/edit-source-buttons.component';
import { isVideoFilePathValid } from './utils';
import { VideoFileFields } from './video-file-fields.component';

interface EditVideoFileProps {
    source: VideoFileSourceType;
    onSaved: () => void;
}

export const EditVideoFile = ({ source, onSaved }: EditVideoFileProps) => {
    const [filePath, setFilePath] = useState<string>(source.config.video_path);
    const updateVideoFileSource = useUpdateSource();

    const isSubmitDisabled =
        updateVideoFileSource.isPending || filePath === source.config.video_path || !isVideoFilePathValid(filePath);

    const handleUpdateVideoFileSource = (active: boolean) => {
        updateVideoFileSource.mutate(
            source.id,
            {
                config: {
                    ...source.config,
                    video_path: filePath.trim(),
                },
                active,
            },
            onSaved
        );
    };

    const handleSave = () => {
        handleUpdateVideoFileSource(source.active);
    };

    const handleSaveAndConnect = () => {
        handleUpdateVideoFileSource(true);
    };

    const handleSubmit = (event: FormEvent) => {
        event.preventDefault();

        handleSave();
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleSubmit}>
            <Flex direction={'column'} gap={'size-200'} marginTop={0}>
                <VideoFileFields filePath={filePath} onFilePathChange={setFilePath} />

                <EditSourceButtons
                    isActiveSource={source.active}
                    onSave={handleSave}
                    onSaveAndConnect={handleSaveAndConnect}
                    isDisabled={isSubmitDisabled}
                    isPending={updateVideoFileSource.isPending}
                />
            </Flex>
        </Form>
    );
};
