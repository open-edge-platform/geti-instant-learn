/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SinkConfig, SinkType } from '@geti-prompt/api';
import { orderBy } from 'lodash-es';

import { ExistingPipelineEntities } from '../../existing-pipeline-entities/existing-pipeline-entities.component';
import { useDeleteSink } from '../api/use-delete-sink';
import { useUpdateSink } from '../api/use-update-sink';
import { MQTTSinkCard } from '../mqtt-sink/mqtt-sink-card.component';
import { isMQTTSink, SinkViews } from '../utils';

interface ExistingSinksListProps {
    sinks: SinkConfig[];
    onViewChange: (view: SinkViews) => void;
    onSetSinkInEditionId: (sinkId: string) => void;
}

const sortSinks = (sinks: SinkConfig[]): SinkConfig[] => {
    return orderBy(sinks, (sink) => sink.active, 'desc');
};

const getMenuItems = ({ isActiveSink }: { isActiveSink: boolean }) => {
    const items = [
        {
            key: 'connect',
            label: 'Connect',
        },
        {
            key: 'edit',
            label: 'Edit',
        },
        {
            key: 'delete',
            label: 'Delete',
        },
    ];

    return items.filter((item) => {
        if (item.key === 'connect' && isActiveSink) {
            return false;
        }

        return true;
    });
};

const ExistingSinksList = ({ sinks, onSetSinkInEditionId, onViewChange }: ExistingSinksListProps) => {
    const deleteSinkMutation = useDeleteSink();
    const updateSinkMutation = useUpdateSink();

    const handleAction = (sink: SinkConfig) => (action: string) => {
        if (action === 'delete') {
            deleteSinkMutation.mutate(sink.id, () => {
                if (sinks.length === 1) {
                    onViewChange('list');
                }
            });
        } else if (action === 'edit') {
            onViewChange('edit');
            onSetSinkInEditionId(sink.id);
        } else if (action === 'connect') {
            updateSinkMutation.mutate({
                sinkId: sink.id,
                body: {
                    config: sink.config,
                    active: true,
                },
            });
        }
    };

    return (
        <ExistingPipelineEntities.List>
            {sortSinks(sinks).map((sink) => {
                if (isMQTTSink(sink)) {
                    return (
                        <MQTTSinkCard
                            key={sink.id}
                            sink={sink}
                            menuItems={getMenuItems({ isActiveSink: sink.active })}
                            onAction={handleAction(sink)}
                        />
                    );
                }

                console.error(
                    `Sink type "${(sink as { config: { sink_type: string } }).config.sink_type}" is not supported.`
                );
                return null;
            })}
        </ExistingPipelineEntities.List>
    );
};

interface ExistingSinksProps {
    sinks: SinkConfig[];
    onViewChange: (view: SinkViews) => void;
    onSetSinkInEditionId: (sinkId: string) => void;
}

const AVAILABLE_SINK_TYPES: SinkType[] = ['mqtt'];

export const ExistingSinks = ({ sinks, onSetSinkInEditionId, onViewChange }: ExistingSinksProps) => {
    const canAddSink = !AVAILABLE_SINK_TYPES.every((sinkType) =>
        sinks.some((sink) => sink.config.sink_type === sinkType)
    );

    return (
        <ExistingPipelineEntities
            addNewEntityButton={
                canAddSink && (
                    <ExistingPipelineEntities.AddNewEntityButton
                        text={'Add new sink'}
                        onPress={() => onViewChange('add')}
                    />
                )
            }
        >
            <ExistingSinksList sinks={sinks} onSetSinkInEditionId={onSetSinkInEditionId} onViewChange={onViewChange} />
        </ExistingPipelineEntities>
    );
};
