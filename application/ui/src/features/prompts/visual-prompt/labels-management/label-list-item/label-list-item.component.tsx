/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { LabelType } from '@/api';
import { useProjectIdentifier } from '@/hooks';
import { ActionButton, Tooltip, TooltipTrigger } from '@geti/ui';
import { Close, Edit } from '@geti/ui/icons';

import { useAnnotationActions } from '../../../../annotator/providers/annotation-actions-provider.component';
import { useVisualPrompt } from '../../visual-prompt-provider.component';
import { useDeleteLabel } from '../api/use-delete-label';
import { useUpdateLabel } from '../api/use-update-label';
import { EditLabel } from '../edit-label/edit-label.component';
import { LabelBadge } from '../label-badge/label-badge.component';

import classes from './label-list-item.module.scss';

interface LabelListItemViewProps {
    label: LabelType;
    onSelect: () => void;
    isSelected: boolean;
    onEdit: () => void;
}

const LabelListItemView = ({ label, onSelect, isSelected, onEdit }: LabelListItemViewProps) => {
    const { projectId } = useProjectIdentifier();
    const deleteLabelMutation = useDeleteLabel();

    const { prompts, setSelectedLabelId } = useVisualPrompt();
    const { deleteAnnotationByLabelId } = useAnnotationActions();

    const deleteLabel = () => {
        deleteLabelMutation.mutate(
            {
                params: {
                    path: {
                        project_id: projectId,
                        label_id: label.id,
                    },
                },
            },
            {
                onSuccess: () => {
                    setSelectedLabelId(null);

                    // If we don't have any prompt, it means that we can safely delete annotations that have assigned
                    //  a given label. If we already have at least one prompt, it's highly probable that prompt has
                    // annotations that include such a label. Deletion of that label would lead to an inconsistent
                    // state. The server should block that case.
                    if (prompts.length === 0) {
                        deleteAnnotationByLabelId(label.id);
                    }
                },
            }
        );
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
    existingLabels: LabelType[];
}

export const LabelListItem = ({ label, onSelect, isSelected, existingLabels }: LabelListItemProps) => {
    const [isInEdition, setIsInEdition] = useState<boolean>(false);
    const { projectId } = useProjectIdentifier();
    const { updateAnnotations, annotations } = useAnnotationActions();

    const updateLabelMutation = useUpdateLabel();
    const updateLabel = (newLabel: LabelType) => {
        updateLabelMutation.mutate(
            {
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
            },
            {
                onSuccess: () => {
                    setIsInEdition(false);

                    if (label.color !== newLabel.color) {
                        const updatedAnnotations = annotations.map((annotation) => ({
                            ...annotation,
                            labels: annotation.labels.map((annotationLabel) =>
                                annotationLabel.id === newLabel.id ? newLabel : annotationLabel
                            ),
                        }));

                        updateAnnotations(updatedAnnotations);
                    }
                },
            }
        );
    };

    const handleClose = () => {
        setIsInEdition(false);
    };

    if (isInEdition) {
        return (
            <EditLabel
                shouldCloseOnOutsideClick
                onAccept={updateLabel}
                onClose={handleClose}
                label={label}
                width={'size-2400'}
                existingLabels={existingLabels}
                isDisabled={updateLabelMutation.isPending}
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
