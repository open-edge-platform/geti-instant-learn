/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Content, Dialog, DialogTrigger } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { EditLabel } from './edit-label.component';
import { Label } from './label.interface';
import { useNewLabel } from './use-new-label.hook';

import classes from './add-label.module.css';

interface AddLabelProps {
    addLabel: (label: Label) => void;
}

export const AddLabel = ({ addLabel }: AddLabelProps) => {
    const { label, resetState } = useNewLabel();

    const handleAddingLabel = (editedLabel: Partial<Label>, closeDialog: () => void) => {
        !isEmpty(editedLabel) && addLabel({ ...label, ...editedLabel } as Label);
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
                            label={label}
                            accept={(editedLabel: Partial<Label>) => handleAddingLabel(editedLabel, _close)}
                            cancel={_close}
                        />
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
