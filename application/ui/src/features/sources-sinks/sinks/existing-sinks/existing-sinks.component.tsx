/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { SinkType } from '@geti-prompt/api';
import { useCurrentProject } from '@geti-prompt/hooks';
import { orderBy } from 'lodash-es';

import { ExistingPipelineEntities } from '../../existing-pipeline-entities/existing-pipeline-entities.component';
import { useDeleteSink } from '../api/use-delete-sink';
import { useUpdateSink } from '../api/use-update-sink';
import { MQTTSinkCard } from '../mqtt-sink/mqtt-sink-card.component';
import { isMQTTSink, SinkViews } from '../utils';

interface ExistingSinksListProps {
    sinks: SinkType[];
    onViewChange: (view: SinkViews) => void;
    onSetSinkInEditionId: (sinkId: string) => void;
}

const sortSinks = (sinks: SinkType[]): SinkType[] => {
    return orderBy(sinks, (sink) => sink.active, 'desc');
};

const getMenuItems = ({ isActiveProject, isActiveSink }: { isActiveProject: boolean; isActiveSink: boolean }) => {
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
        if (item.key === 'edit' && !isActiveProject) {
            return false;
        }

        return true;
    });
};

const ExistingSinksList = ({ sinks, onSetSinkInEditionId, onViewChange }: ExistingSinksListProps) => {
    const { data: project } = useCurrentProject();
    const isActiveProject = project.active;

    const deleteSinkMutation = useDeleteSink();
    const updateSinkMutation = useUpdateSink();

    const handleAction = (sink: SinkType) => (action: string) => {
        if (action === 'delete') {
            deleteSinkMutation.mutate(
                {
                    params: {
                        path: {
                            project_id: project.id,
                            sink_id: sink.id,
                        },
                    },
                },
                {
                    onSuccess: () => {
                        if (sinks.length === 1) {
                            onViewChange('list');
                        }
                    },
                }
            );
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
                            menuItems={getMenuItems({ isActiveSink: sink.active, isActiveProject })}
                            onAction={handleAction(sink)}
                        />
                    );
                }

                throw new Error(
                    `Sink type "${(sink as { config: { sink_type: string } }).config.sink_type}" is not 
                    supported.`
                );
            })}
        </ExistingPipelineEntities.List>
    );
};

interface ExistingSinksProps {
    sinks: SinkType[];
    onViewChange: (view: SinkViews) => void;
    onSetSinkInEditionId: (sinkId: string) => void;
}

export const ExistingSinks = ({ sinks, onSetSinkInEditionId, onViewChange }: ExistingSinksProps) => {
    return (
        <ExistingPipelineEntities
            addNewEntityButton={
                <ExistingPipelineEntities.AddNewEntityButton
                    text={'Add new sink'}
                    onPress={() => onViewChange('add')}
                />
            }
        >
            <ExistingSinksList sinks={sinks} onSetSinkInEditionId={onSetSinkInEditionId} onViewChange={onViewChange} />
        </ExistingPipelineEntities>
    );
};
