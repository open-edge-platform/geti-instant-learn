/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useMemo } from 'react';

import { $api, LabelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Content, Dialog } from '@geti/ui';
import { getDistinctColorBasedOnHash } from '@geti/ui/utils';
import { v4 as uuid } from 'uuid';

import { EditLabel } from '../edit-label/edit-label.component';

interface AddLabelDialogProps {
    closeDialog: () => void;
    existingLabels: LabelType[];
}

const useAddLabel = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('post', '/api/v1/projects/{project_id}/labels', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/labels', { params: { path: { project_id: projectId } } }],
            ],
        },
    });
};

const getDefaultLabel = (): LabelType => {
    const id = uuid();

    return {
        id,
        name: '',
        color: getDistinctColorBasedOnHash(id),
    };
};

export const AddLabelDialog = ({ existingLabels, closeDialog }: AddLabelDialogProps) => {
    const { projectId } = useProjectIdentifier();

    const defaultLabel = useMemo(getDefaultLabel, []);

    const addLabelMutation = useAddLabel();

    const addLabel = (label: LabelType) => {
        addLabelMutation.mutate(
            {
                body: {
                    id: label.id,
                    name: label.name,
                    color: label.color,
                },
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            },
            {
                onSuccess: closeDialog,
            }
        );
    };

    return (
        <Dialog>
            <Content>
                <EditLabel
                    label={defaultLabel}
                    onAccept={addLabel}
                    onClose={closeDialog}
                    isDisabled={addLabelMutation.isPending}
                    existingLabels={existingLabels}
                />
            </Content>
        </Dialog>
    );
};
