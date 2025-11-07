/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, ButtonProps } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useAnnotationActions } from '../../annotator/providers/annotation-actions-provider.component';
import { useSavePrompt } from './api/use-save-prompt';

interface SavePromptProps {
    frameId: string;
    justifySelf?: ButtonProps['justifySelf'];
}

export const SavePrompt = ({ frameId, justifySelf }: SavePromptProps) => {
    const savePrompt = useSavePrompt();
    const { annotations } = useAnnotationActions();

    const isSavePromptDisabled = isEmpty(annotations) || savePrompt.isPending;

    const handleSavePrompt = () => {
        savePrompt.mutate(frameId);
    };

    return (
        <Button
            justifySelf={justifySelf}
            variant={'secondary'}
            isDisabled={isSavePromptDisabled}
            isPending={savePrompt.isPending}
            onPress={handleSavePrompt}
        >
            Save prompt
        </Button>
    );
};
