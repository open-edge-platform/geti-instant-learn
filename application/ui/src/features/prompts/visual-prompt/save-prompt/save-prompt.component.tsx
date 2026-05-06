/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useAnnotationActions } from '../../../annotator/providers/annotation-actions-provider.component';
import { useModelStatus } from '../../../model-status';
import { useSavePrompt } from '../api/use-save-prompt';
import { useVisualPrompt } from '../visual-prompt-provider.component';

export const SavePrompt = () => {
    const savePrompt = useSavePrompt();
    const { annotations } = useAnnotationActions();
    const { selectedLabelId } = useVisualPrompt();
    const { isBusy } = useModelStatus();

    const isSavePromptDisabled = isEmpty(annotations) || savePrompt.isPending || selectedLabelId === null || isBusy;

    return (
        <Button
            alignSelf={'end'}
            variant={'secondary'}
            isDisabled={isSavePromptDisabled}
            isPending={savePrompt.isPending}
            onPress={savePrompt.mutate}
        >
            Save prompt
        </Button>
    );
};
