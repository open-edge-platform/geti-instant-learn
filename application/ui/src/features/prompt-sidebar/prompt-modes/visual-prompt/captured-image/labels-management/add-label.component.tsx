/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { Button, Content, Dialog, DialogTrigger } from '@geti/ui';
import { isEmpty } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { EditLabel } from './edit-label.component';
import { Label } from './label.interface';
import { getDistinctColorBasedOnHash } from './utils-temp';

import classes from './add-label.module.css';

interface AddLabelProps {
    addLabel: (label: Label) => void;
}

export const AddLabel = ({ addLabel }: AddLabelProps) => {
    //TODO: hook???
    const getBaseNewLabel = (): Label => {
        const newId = uuid();
        return { id: newId, name: '', color: getDistinctColorBasedOnHash(newId) };
    };

    const [newLabel, setNewLabel] = useState<Label>(getBaseNewLabel());

    const resetState = () => {
        setNewLabel(getBaseNewLabel());
    };

    //-------------------------
    const handleAddingLabel = (editedLabel: Partial<Label>, closeDialog: () => void) => {
        !isEmpty(editedLabel) && addLabel({ ...newLabel, ...editedLabel } as Label);
        resetState();
        closeDialog();
    };

    const onDialogClose = (isOpen: boolean) => {
        if (!isOpen) resetState();
    };

    return (
        <DialogTrigger type={'popover'} hideArrow placement={'bottom right'} onOpenChange={onDialogClose}>
            <Button variant={'secondary'} UNSAFE_style={{ border: 'none' }} UNSAFE_className={classes.addLabelButton}>
                Add label
            </Button>
            {(_close) => (
                <Dialog>
                    <Content>
                        <EditLabel
                            label={newLabel}
                            accept={(editedLabel: Partial<Label>) => handleAddingLabel(editedLabel, _close)}
                            cancel={_close}
                        />
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
