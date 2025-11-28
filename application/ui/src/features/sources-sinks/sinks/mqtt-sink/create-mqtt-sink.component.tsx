/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Checkbox, CheckboxGroup, Flex, Form, NumberField, TextField, View } from '@geti/ui';

import { MQTTSinkFields } from './mqtt-sink-fields.component';

interface CreateMQTTSinkProps {
    onSaved: () => void;
}

export const CreateMQTTSink = ({ onSaved }: CreateMQTTSinkProps) => {
    return (
        <View>
            <Form>
                <MQTTSinkFields />

                <Button width={'max-content'}>Apply</Button>
            </Form>
        </View>
    );
};
