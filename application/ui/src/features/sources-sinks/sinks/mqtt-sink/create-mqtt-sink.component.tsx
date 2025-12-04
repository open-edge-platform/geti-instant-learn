/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent } from 'react';

import { Button, Form } from '@geti/ui';

import { useCreateSink } from '../api/use-create-sink';
import { MQTTSinkFields } from './mqtt-sink-fields.component';

interface CreateMQTTSinkProps {
    onSaved: () => void;
}

export const CreateMQTTSink = ({ onSaved }: CreateMQTTSinkProps) => {
    const createSinkMutation = useCreateSink();

    const createSink = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const formData = Object.fromEntries(new FormData(event.currentTarget));

        createSinkMutation.mutate(
            {
                sink_type: 'mqtt',
                name: formData.name.toString(),
                broker_port: Number(formData.broker_port),
                broker_host: formData.broker_host.toString(),
                topic: formData.topic.toString(),
                auth_required: formData.auth_required === 'on',
            },
            onSaved
        );
    };

    return (
        <Form validationBehavior={'native'} onSubmit={createSink}>
            <MQTTSinkFields />

            <Button type={'submit'} width={'max-content'} isPending={createSinkMutation.isPending}>
                Apply
            </Button>
        </Form>
    );
};
