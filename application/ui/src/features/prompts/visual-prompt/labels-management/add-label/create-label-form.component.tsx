/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useMemo, useState } from 'react';

import { LabelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { Flex } from '@geti/ui';
import { Add } from '@geti/ui/icons';
import { getDistinctColorBasedOnHash } from '@geti/ui/utils';
import { isEmpty } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { useVisualPrompt } from '../../visual-prompt-provider.component';
import { useCreateLabel } from '../api/use-create-label';
import { Label } from '../label/label.component';
import { validateLabelName } from '../utils';

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
    label: LabelType;
    onClose: () => void;
    existingLabels: LabelType[];
    isDisabled?: boolean;
}

const CreateLabel = ({ onCreateLabel, label, onClose, existingLabels, isDisabled }: CreateLabelProps) => {
    const [color, setColor] = useState<string>(label.color);
    const [name, setName] = useState<string>(label.name);

    const validationError = validateLabelName(name, existingLabels, label.id);
    const isCreateButtonDisabled = !!validationError || isDisabled || isEmpty(name.trim());

    const handleCreateLabel = () => {
        onCreateLabel({ color, name, id: label.id });
    };

    return (
        <Label.Form onSubmit={handleCreateLabel}>
            <Flex marginTop={0} gap={'size-50'} justifyContent={'center'}>
                <Label.ColorPicker color={color} onColorChange={setColor} />

                <Label.NameField
                    name={name}
                    onChangeName={setName}
                    onClose={onClose}
                    ariaLabel={'New label name'}
                    validationError={validationError}
                />

                <Label.Button isDisabled={isCreateButtonDisabled} color={'var(--spectrum-global-color-gray-200)'}>
                    <Add />
                </Label.Button>
            </Flex>
        </Label.Form>
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
        <CreateLabel
            existingLabels={existingLabels}
            label={defaultLabel}
            onCreateLabel={addLabel}
            onClose={onClose}
            isDisabled={createLabelMutation.isPending}
        />
    );
};
