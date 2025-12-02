/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useMemo } from 'react';

import { LabelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Content, Dialog } from '@geti/ui';
import { getDistinctColorBasedOnHash } from '@geti/ui/utils';
import { v4 as uuid } from 'uuid';

import { useVisualPrompt } from '../../visual-prompt-provider.component';
import { useCreateLabel } from '../api/use-create-label';
import { EditLabel } from '../edit-label/edit-label.component';

interface AddLabelDialogProps {
    closeDialog: () => void;
    existingLabels: LabelType[];
}

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

    const { selectedLabelId, setSelectedLabelId } = useVisualPrompt();
    const defaultLabel = useMemo(getDefaultLabel, []);

    const createLabelMutation = useCreateLabel();

    const addLabel = (label: LabelType) => {
        createLabelMutation.mutate(
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
                onSuccess: () => {
                    if (selectedLabelId === null) {
                        setSelectedLabelId(label.id);
                    }
                    closeDialog();
                },
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
                    isDisabled={createLabelMutation.isPending}
                    existingLabels={existingLabels}
                />
            </Content>
        </Dialog>
    );
};
