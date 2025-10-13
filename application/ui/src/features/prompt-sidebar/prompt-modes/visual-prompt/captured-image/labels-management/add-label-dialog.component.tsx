/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, Dialog } from '@geti/ui';
import { getDistinctColorBasedOnHash } from '@geti/ui/utils';
import { v4 as uuid } from 'uuid';

import { EditLabel } from './edit-label.component';
import { Label } from './label.interface';

interface AddLabelDialogProps {
    onAction: (label: Label) => void;
    closeDialog: () => void;
}

export const AddLabelDialog = ({ onAction, closeDialog }: AddLabelDialogProps) => {
    const id = uuid();
    const defaultLabel = { id, name: '', color: getDistinctColorBasedOnHash(id) };

    return (
        <Dialog>
            <Content>
                <EditLabel label={defaultLabel} onAccept={onAction} onCancel={closeDialog} />
            </Content>
        </Dialog>
    );
};
