/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useRef } from 'react';

import { MQTTSinkType } from '@geti-prompt/api';
import { Button, ButtonGroup, Form } from '@geti/ui';

import { useUpdateSink } from '../api/use-update-sink';
import { MQTTSinkFields } from './mqtt-sink-fields.component';

interface EditMQTTSinkProps {
    sink: MQTTSinkType;
    onSaved: () => void;
}

export const EditMQTTSink = ({ sink, onSaved }: EditMQTTSinkProps) => {
    const updateSinkMutation = useUpdateSink();
    const activeRef = useRef(sink.active);

    const editSink = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const formData = Object.fromEntries(new FormData(event.currentTarget));

        updateSinkMutation.mutate(
            {
                sinkId: sink.id,
                body: {
                    config: {
                        sink_type: 'mqtt',
                        broker_port: Number(formData.broker_port),
                        broker_host: formData.broker_host.toString(),
                        topic: formData.topic.toString(),
                        auth_required: formData.auth_required.toString() === 'on',
                    },
                    active: activeRef.current,
                },
            },
            onSaved
        );
    };

    return (
        <Form onSubmit={editSink}>
            <MQTTSinkFields />

            <ButtonGroup>
                <Button type={'submit'} onPress={() => (activeRef.current = sink.active)}>
                    Save
                </Button>
                {!sink.active && (
                    <Button type={'submit'} onPress={() => (activeRef.current = true)}>
                        Save & Connect
                    </Button>
                )}
            </ButtonGroup>
        </Form>
    );
};
