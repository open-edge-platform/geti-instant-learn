/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { MQTT } from '@geti-prompt/icons';

import { DisclosureGroup } from '../disclosure-group/disclosure-group.component';
import { PipelineEntityPanel } from '../pipeline-entity-panel/pipeline-entity-panel.component';
import { useSinks } from './api/use-sinks';
import { CreateMQTTSink } from './mqtt-sink/create-mqtt-sink.component';
import { SinkViews } from './utils';

interface SinksListProps {
    onViewChange: (view: SinkViews) => void;
}

const SinksList = ({ onViewChange }: SinksListProps) => {
    const items = [
        {
            label: 'MQTT',
            icon: <MQTT width={'24px'} />,
            value: 'mqtt',
            content: <CreateMQTTSink onSaved={() => onViewChange('list')} />,
        },
    ];

    return <DisclosureGroup items={items} value={'mqtt'} />;
};

interface AddSinkProps {
    onViewChange: (view: SinkViews) => void;
}

const AddSink = ({ onViewChange }: AddSinkProps) => {
    return (
        <PipelineEntityPanel
            title={<PipelineEntityPanel.Title>Add new output sink</PipelineEntityPanel.Title>}
            onBackClick={() => onViewChange('existing')}
        >
            <SinksList onViewChange={onViewChange} />
        </PipelineEntityPanel>
    );
};

export const Sinks = () => {
    const { data } = useSinks();

    const [view, setView] = useState<SinkViews>(data.sinks.length === 0 ? 'list' : 'existing');

    if (view === 'add') {
        return <AddSink onViewChange={setView} />;
    }

    return <SinksList onViewChange={setView} />;
};
