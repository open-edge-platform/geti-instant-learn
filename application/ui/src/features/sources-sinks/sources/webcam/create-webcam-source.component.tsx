/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Flex, Form } from '@geti/ui';

import { useCreateSource } from '../api/use-create-source';
import { isDeviceIdValid } from './utils';
import { WebcamSourceFields } from './webcam-source-fields.component';

interface CreateWebcamSourceProps {
    onSaved: () => void;
}

export const CreateWebcamSource = ({ onSaved }: CreateWebcamSourceProps) => {
    const [selectedDeviceId, setSelectedDeviceId] = useState<string>('0');
    const createWebcamSource = useCreateSource();

    const isApplyDisabled = !isDeviceIdValid(selectedDeviceId) || createWebcamSource.isPending;

    const handleApply = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        createWebcamSource.mutate(
            {
                source_type: 'webcam',
                device_id: parseInt(selectedDeviceId),
                seekable: false,
            },
            onSaved
        );
    };

    return (
        <Form validationBehavior={'native'} onSubmit={handleApply}>
            <Flex direction={'column'} gap={'size-200'} marginTop={0}>
                <WebcamSourceFields selectedDeviceId={selectedDeviceId} onSetSelectedDeviceId={setSelectedDeviceId} />

                <ButtonGroup>
                    <Button type={'submit'} isPending={createWebcamSource.isPending} isDisabled={isApplyDisabled}>
                        Apply
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
