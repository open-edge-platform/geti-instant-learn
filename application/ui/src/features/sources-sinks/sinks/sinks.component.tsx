/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode, useState } from 'react';

import { SinkConfig, SinkType } from '@geti-prompt/api';
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
    sinks: SinkConfig[];
}

const SinksList = ({ onViewChange, sinks }: SinksListProps) => {
    const sinksList = [
        {
            label: 'MQTT',
            icon: <MQTT width={'24px'} />,
            value: 'mqtt',
            content: <CreateMQTTSink onSaved={() => onViewChange('existing')} />,
        },
    ] satisfies { label: string; icon: ReactNode; value: SinkType; content: ReactNode }[];

    const items = sinksList.filter(
        (sink) => !sinks.some((existingSink) => existingSink.config.sink_type === sink.value)
    );

    return <DisclosureGroup items={items} value={'mqtt'} />;
};

interface AddSinkProps {
    onViewChange: (view: SinkViews) => void;
    sinks: SinkConfig[];
}

const AddSink = ({ onViewChange, sinks }: AddSinkProps) => {
    return (
        <PipelineEntityPanel
            title={<PipelineEntityPanel.Title>Add new output sink</PipelineEntityPanel.Title>}
            onBackClick={() => onViewChange('existing')}
        >
            <SinksList onViewChange={onViewChange} sinks={sinks} />
        </PipelineEntityPanel>
    );
};

export const Sinks = () => {
    const {
        data: { sinks },
    } = useSinks();

    const [sinkInEditionId, setSinkInEditionId] = useState<string | null>(null);
    const sinkInEdition = sinks.find((sink) => sink.id === sinkInEditionId);

    const [view, setView] = useState<SinkViews>(sinks.length === 0 ? 'list' : 'existing');

    if (view === 'add') {
        return <AddSink onViewChange={setView} sinks={sinks} />;
    }

    if (view === 'existing') {
        return <ExistingSinks sinks={sinks} onViewChange={setView} onSetSinkInEditionId={setSinkInEditionId} />;
    }

    if (view === 'edit' && sinkInEdition !== undefined) {
        return <EditSink sink={sinkInEdition} onViewChange={setView} />;
    }

    return <SinksList onViewChange={setView} sinks={sinks} />;
};
