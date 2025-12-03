/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { SinkType } from '@geti-prompt/api';

import { PipelineEntityPanel } from '../../pipeline-entity-panel/pipeline-entity-panel.component';
import { EditMQTTSink } from '../mqtt-sink/edit-mqtt-sink.component';
import { isMQTTSink, SinkViews } from '../utils';

interface EditSinkContainerProps {
    onBackClick: () => void;
    children: ReactNode;
    title: string;
}

const EditSinkContainer = ({ title, children, onBackClick }: EditSinkContainerProps) => {
    return (
        <PipelineEntityPanel
            title={<PipelineEntityPanel.Title>Edit output sink</PipelineEntityPanel.Title>}
            onBackClick={onBackClick}
        >
            <PipelineEntityPanel.Content title={title}>{children}</PipelineEntityPanel.Content>
        </PipelineEntityPanel>
    );
};

interface EditSinkProps {
    sink: SinkType;
    onViewChange: (view: SinkViews) => void;
}

export const EditSink = ({ sink, onViewChange }: EditSinkProps) => {
    const handleGoBack = () => onViewChange('existing');

    if (isMQTTSink(sink)) {
        return (
            <EditSinkContainer onBackClick={handleGoBack} title={'MQTT'}>
                <EditMQTTSink sink={sink} onSaved={handleGoBack} />
            </EditSinkContainer>
        );
    }

    throw new Error(`Sink type "${(sink as { config: { sink_type: string } }).config.sink_type}" is not supported.`);
};
