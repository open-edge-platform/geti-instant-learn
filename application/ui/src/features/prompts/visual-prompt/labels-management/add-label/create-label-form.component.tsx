/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useMemo } from 'react';

import { LabelType } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { Flex } from '@geti/ui';
import { Add } from '@geti/ui/icons';
import { getDistinctColorBasedOnHash } from '@geti/ui/utils';
import { v4 as uuid } from 'uuid';

import { useVisualPrompt } from '../../visual-prompt-provider.component';
import { useCreateLabel } from '../api/use-create-label';
import { Label } from '../label/label.component';

const getDefaultLabel = (): LabelType => {
    const id = uuid();

    return {
        id,
        name: '',
        color: getDistinctColorBasedOnHash(id),
    };
};

interface CreateLabelProps {
    onCreateLabel: (newLabel: LabelType) => void;
    onClose: () => void;
    existingLabels: LabelType[];
    isDisabled?: boolean;
}

const CreateLabel = ({ onCreateLabel, onClose, existingLabels, isDisabled }: CreateLabelProps) => {
    const defaultLabel = useMemo(() => getDefaultLabel(), []);

    return (
        <Label label={defaultLabel} existingLabels={existingLabels}>
            <Label.Form onSubmit={onCreateLabel}>
                <Flex marginTop={0} gap={'size-50'} justifyContent={'center'}>
                    <Label.ColorPicker />

                    <Label.NameField onClose={onClose} ariaLabel={'New label name'} />

                    <Label.Button isDisabled={isDisabled} color={'var(--spectrum-global-color-gray-200)'}>
                        <Add />
                    </Label.Button>
                </Flex>
            </Label.Form>
        </Label>
    );
};

interface CreateLabelFormProps {
    onClose: () => void;
    onSuccess?: (label: LabelType) => void;
    existingLabels?: LabelType[];
}

export const CreateLabelForm = ({ onClose, existingLabels = [], onSuccess }: CreateLabelFormProps) => {
    const { projectId } = useProjectIdentifier();

    const { selectedLabelId, setSelectedLabelId } = useVisualPrompt();

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
        <CreateLabel
            existingLabels={existingLabels}
            onCreateLabel={addLabel}
            onClose={onClose}
            isDisabled={createLabelMutation.isPending}
        />
    );
};
