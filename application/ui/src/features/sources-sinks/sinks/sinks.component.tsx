/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { MQTT } from '@geti-prompt/icons';

import { DisclosureGroup } from '../disclosure-group/disclosure-group.component';
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

export const Sinks = () => {
    const { data } = useSinks();

    const [view, setView] = useState<SinkViews>('list');

    return <SinksList onViewChange={setView} />;
};
