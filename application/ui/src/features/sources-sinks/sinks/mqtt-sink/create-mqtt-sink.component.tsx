/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { MQTTSinkType } from '@geti-prompt/api';
import { Button, Form } from '@geti/ui';

import { useCreateSink } from '../api/use-create-sink';
import { MQTTSinkFields } from './mqtt-sink-fields.component';

interface CreateMQTTSinkProps {
    onSaved: () => void;
}

export const CreateMQTTSink = ({ onSaved }: CreateMQTTSinkProps) => {
    const createSinkMutation = useCreateSink();
    const [sinkConfig, setSinkConfig] = useState<MQTTSinkType['config']>({
        sink_type: 'mqtt',
        name: '',
        broker_port: 0,
        broker_host: '',
        topic: '',
        auth_required: false,
    });

    const createSink = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        createSinkMutation.mutate(sinkConfig, onSaved);
    };

    return (
        <Form validationBehavior={'native'} onSubmit={createSink}>
            <MQTTSinkFields sinkConfig={sinkConfig} onSetSinkConfig={setSinkConfig} />

            <Button type={'submit'} width={'max-content'} isPending={createSinkMutation.isPending}>
                Apply
            </Button>
        </Form>
    );
};
