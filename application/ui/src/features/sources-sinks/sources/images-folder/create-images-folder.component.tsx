/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Flex, Form } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';
import { ImagesFolderFields } from './images-folder-fields.component';
import { isFolderPathValid } from './utils';

interface CreateImagesFolderProps {
    onSaved: () => void;
}

export const CreateImagesFolder = ({ onSaved }: CreateImagesFolderProps) => {
    const [folderPath, setFolderPath] = useState<string>('');
    const createImagesFolderSource = useCreateSource();

    const isApplyDisabled = !isFolderPathValid(folderPath) || createImagesFolderSource.isPending;

    const submit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        createImagesFolderSource.mutate(
            {
                source_type: 'images_folder',
                images_folder_path: folderPath.trim(),
                seekable: true,
            },
            onSaved
        );
    };

    return (
        <Form validationBehavior={'native'} onSubmit={submit}>
            <Flex gap={'size-200'} direction={'column'} marginTop={0}>
                <ImagesFolderFields folderPath={folderPath} onSetFolderPath={setFolderPath} />

                <ButtonGroup>
                    <Button
                        type={'submit'}
                        variant={'accent'}
                        isDisabled={isApplyDisabled}
                        isPending={createImagesFolderSource.isPending}
                    >
                        Apply
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
