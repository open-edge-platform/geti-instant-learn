/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useMemo } from 'react';

import { LabelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { getDistinctColorBasedOnHash } from '@geti/ui/utils';
import { v4 as uuid } from 'uuid';

import { useVisualPrompt } from '../../visual-prompt-provider.component';
import { useCreateLabel } from '../api/use-create-label';
import { EditLabel } from '../edit-label/edit-label.component';

const getDefaultLabel = (): LabelType => {
    const id = uuid();

    return {
        id,
        name: '',
        color: getDistinctColorBasedOnHash(id),
    };
};

interface CreateLabelFormProps {
    onClose: () => void;
    onSuccess?: (label: LabelType) => void;
    existingLabels?: LabelType[];
}

export const CreateLabelForm = ({ onClose, existingLabels = [], onSuccess }: CreateLabelFormProps) => {
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
                onSuccess: async () => {
                    if (selectedLabelId === null) {
                        setSelectedLabelId(label.id);
                    }

                    onSuccess?.(label);

                    onClose();
                },
            }
        );
    };

    return (
        <EditLabel
            label={defaultLabel}
            onAccept={addLabel}
            onClose={onClose}
            isDisabled={createLabelMutation.isPending}
            existingLabels={existingLabels}
        />
    );
};
