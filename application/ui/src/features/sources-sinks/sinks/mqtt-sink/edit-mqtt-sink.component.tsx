/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useRef, useState } from 'react';

import { MQTTSinkType } from '@geti-prompt/api';
import { Button, ButtonGroup, Form } from '@geti/ui';
import { isEqual } from 'lodash-es';

import { useUpdateSink } from '../api/use-update-sink';
import { MQTTSinkFields } from './mqtt-sink-fields.component';

interface EditMQTTSinkProps {
    sink: MQTTSinkType;
    onSaved: () => void;
}

export const EditMQTTSink = ({ sink, onSaved }: EditMQTTSinkProps) => {
    const updateSinkMutation = useUpdateSink();
    const activeRef = useRef(sink.active);
    const [sinkConfig, setSinkConfig] = useState<MQTTSinkType['config']>(sink.config);

    const isSaveDisabled = updateSinkMutation.isPending || isEqual(sinkConfig, sink.config);

    const editSink = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        updateSinkMutation.mutate(
            {
                sinkId: sink.id,
                body: {
                    config: sinkConfig,
                    active: activeRef.current,
                },
            },
            onSaved
        );
    };

    return (
        <Form validationBehavior={'native'} onSubmit={editSink}>
            <MQTTSinkFields sinkConfig={sinkConfig} onSetSinkConfig={setSinkConfig} />

            <ButtonGroup>
                <Button
                    type={'submit'}
                    onPress={() => (activeRef.current = sink.active)}
                    isPending={updateSinkMutation.isPending}
                    isDisabled={isSaveDisabled}
                >
                    Save
                </Button>
                {!sink.active && (
                    <Button
                        type={'submit'}
                        onPress={() => (activeRef.current = true)}
                        isPending={updateSinkMutation.isPending}
                        isDisabled={isSaveDisabled}
                    >
                        Save & Connect
                    </Button>
                )}
            </ButtonGroup>
        </Form>
    );
};
