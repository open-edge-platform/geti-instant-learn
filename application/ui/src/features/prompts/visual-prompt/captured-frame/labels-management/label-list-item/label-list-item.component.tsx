/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { $api, LabelType } from '@geti-prompt/api';
import { useProjectIdentifier } from '@geti-prompt/hooks';
import { ActionButton, Tooltip, TooltipTrigger } from '@geti/ui';
import { Close, Edit } from '@geti/ui/icons';

import { EditLabel } from '../edit-label/edit-label.component';
import { LabelBadge } from '../label-badge/label-badge.component';

import classes from './label-list-item.module.scss';

interface LabelListItemViewProps {
    label: LabelType;
    onSelect: () => void;
    isSelected: boolean;
    onEdit: () => void;
}

const useDeleteLabelMutation = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('delete', '/api/v1/projects/{project_id}/labels/{label_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/labels', { params: { path: { project_id: projectId } } }],
            ],
        },
    });
};

/*
const useUpdateLabelMutation = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('put', '/api/v1/projects/{project_id}/labels/{label_id}', {
        meta: {
            invalidates: [
                ['get', '/api/v1/projects/{project_id}/labels', { params: { path: { project_id: projectId } } }],
            ],
        },
    });
};
*/

const LabelListItemView = ({ label, onSelect, isSelected, onEdit }: LabelListItemViewProps) => {
    const { projectId } = useProjectIdentifier();
    const deleteLabelMutation = useDeleteLabelMutation();

    const deleteLabel = () => {
        deleteLabelMutation.mutate({
            params: {
                path: {
                    project_id: projectId,
                    label_id: label.id,
                },
            },
        });
    };

    return (
        <LabelBadge onClick={onSelect} key={label.id} label={label} isSelected={isSelected}>
            <TooltipTrigger placement={'bottom'}>
                <ActionButton
                    aria-label={`Edit ${label.name} label`}
                    isQuiet
                    UNSAFE_className={classes.iconButton}
                    onPress={onEdit}
                >
                    <Edit />
                </ActionButton>
                <Tooltip>Edit label name</Tooltip>
            </TooltipTrigger>
            <TooltipTrigger placement={'bottom'}>
                <ActionButton
                    aria-label={`Delete ${label.name} label`}
                    isQuiet
                    UNSAFE_className={classes.iconButton}
                    onPress={deleteLabel}
                >
                    <Close />
                </ActionButton>
                <Tooltip>Delete label</Tooltip>
            </TooltipTrigger>
        </LabelBadge>
    );
};

interface LabelListItemProps {
    label: LabelType;
    onSelect: () => void;
    isSelected: boolean;
    existingLabelsNames: string[];
}

export const LabelListItem = ({ label, onSelect, isSelected, existingLabelsNames }: LabelListItemProps) => {
    const [isInEdition, setIsInEdition] = useState<boolean>(false);
    /*const { projectId } = useProjectIdentifier();

    const updateLabelMutation = useUpdateLabelMutation();
    const updateLabel = (newLabel: LabelType) => {
        updateLabelMutation.mutate({
            body: {
                color: newLabel.color,
                name: newLabel.name,
            },
            params: {
                path: {
                    project_id: projectId,
                    label_id: label.id,
                },
            },
        });
    };*/

    if (isInEdition) {
        return (
            <EditLabel
                //onAccept={updateLabel}
                onAccept={() => {}}
                onClose={() => setIsInEdition(false)}
                label={label}
                isQuiet
                width={'size-2400'}
                existingLabelsNames={existingLabelsNames}
                //isDisabled={updateLabelMutation.isPending}
            />
        );
    }

    return (
        <LabelListItemView
            label={label}
            onSelect={onSelect}
            isSelected={isSelected}
            onEdit={() => setIsInEdition(true)}
        />
    );
};
