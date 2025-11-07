/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useAnnotationActions } from '../../annotator/providers/annotation-actions-provider.component';
import { useSavePrompt } from './api/use-save-prompt';

export const SavePrompt = () => {
    const savePrompt = useSavePrompt();
    const { annotations } = useAnnotationActions();

    const isSavePromptDisabled = isEmpty(annotations) || savePrompt.isPending;

    return (
        <Button
            justifySelf={'end'}
            variant={'secondary'}
            isDisabled={isSavePromptDisabled}
            isPending={savePrompt.isPending}
            onPress={savePrompt.mutate}
        >
            Save prompt
        </Button>
    );
};
