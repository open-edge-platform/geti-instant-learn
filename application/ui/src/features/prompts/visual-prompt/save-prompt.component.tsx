/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useAnnotationActions } from '../../annotator/providers/annotation-actions-provider.component';
import { useAnnotator } from '../../annotator/providers/annotator-provider.component';
import { useSavePrompt } from './api/use-save-prompt';

export const SavePrompt = () => {
    const savePrompt = useSavePrompt();
    const { annotations } = useAnnotationActions();
    const { frameId } = useAnnotator();

    const isSavePromptDisabled = isEmpty(annotations) || savePrompt.isPending;

    const handleSavePrompt = () => {
        savePrompt.mutate(frameId);
    };

    return (
        <Button
            justifySelf={'end'}
            variant={'secondary'}
            isDisabled={isSavePromptDisabled}
            isPending={savePrompt.isPending}
            onPress={handleSavePrompt}
        >
            Save prompt
        </Button>
    );
};
