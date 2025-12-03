/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { SinkType } from '@geti-prompt/api';
import { MQTT } from '@geti-prompt/icons';

import { DisclosureGroup } from '../disclosure-group/disclosure-group.component';
import { PipelineEntityPanel } from '../pipeline-entity-panel/pipeline-entity-panel.component';
import { useSinks } from './api/use-sinks';
import { EditSink } from './edit-sinks/edit-sink.component';
import { ExistingSinks } from './existing-sinks/existing-sinks.component';
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
    const {
        data: { sinks: sinks2 },
    } = useSinks();
    const sinks: SinkType[] = [
        {
            id: '1',
            name: 'MQTT',
            active: true,
            config: {
                sink_type: 'mqtt',
                name: 'MQTT',
                broker_host: 'localhost',
                topic: 'test',
                broker_port: 1883,
                auth_required: false,
            },
        },
    ];
    const [sinkInEditionId, setSinkInEditionId] = useState<string | null>(null);
    const sinkInEdition = sinks.find((sink) => sink.id === sinkInEditionId);

    const [view, setView] = useState<SinkViews>(sinks.length === 0 ? 'list' : 'existing');

    if (view === 'add') {
        return <AddSink onViewChange={setView} />;
    }

    if (view === 'existing') {
        return <ExistingSinks sinks={sinks} onViewChange={setView} onSetSinkInEditionId={setSinkInEditionId} />;
    }

    if (view === 'edit' && sinkInEdition !== undefined) {
        return <EditSink sink={sinkInEdition} onViewChange={setView} />;
    }

    return <SinksList onViewChange={setView} />;
};
