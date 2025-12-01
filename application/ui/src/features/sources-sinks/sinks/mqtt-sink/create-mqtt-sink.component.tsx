/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Form } from '@geti/ui';

import { MQTTSinkFields } from './mqtt-sink-fields.component';

interface CreateMQTTSinkProps {
    onSaved: () => void;
}

export const CreateMQTTSink = ({ onSaved }: CreateMQTTSinkProps) => {
    const createSink = () => {
        //createSinkMutation.mutate({})
        onSaved();
    };

    return (
        <Form>
            <MQTTSinkFields />

            <Button width={'max-content'} onPress={createSink}>
                Apply
            </Button>
        </Form>
    );
};
