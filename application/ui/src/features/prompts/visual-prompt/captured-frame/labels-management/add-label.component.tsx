/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, DialogTrigger } from '@geti/ui';

import { Label } from '../../../../annotator/types';
import { AddLabelDialog } from './add-label-dialog.component';

import classes from './add-label.module.scss';

interface AddLabelProps {
    onAddLabel: (label: Label) => void;
}

export const AddLabel = ({ onAddLabel }: AddLabelProps) => {
    return (
        <DialogTrigger type={'popover'} hideArrow placement={'bottom right'}>
            <Button variant={'secondary'} UNSAFE_className={classes.addLabelButton}>
                Add label
            </Button>
            {(close) => <AddLabelDialog onAction={onAddLabel} closeDialog={close} />}
        </DialogTrigger>
    );
};
