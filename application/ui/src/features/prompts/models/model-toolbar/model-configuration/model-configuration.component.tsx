/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelType } from '@geti-prompt/api';
import { Cog } from '@geti-prompt/icons';
import { ActionButton, DialogTrigger } from '@geti/ui';

import { ModelConfigurationDialog } from './model-configuration-dialog.component';

interface ModelConfigurationProps {
    model: ModelType;
}

export const ModelConfiguration = ({ model }: ModelConfigurationProps) => {
    return (
        <DialogTrigger type={'popover'}>
            <ActionButton isQuiet aria-label={'Configure model'}>
                <Cog />
            </ActionButton>
            {(close) => <ModelConfigurationDialog model={model} onClose={close} />}
        </DialogTrigger>
    );
};
