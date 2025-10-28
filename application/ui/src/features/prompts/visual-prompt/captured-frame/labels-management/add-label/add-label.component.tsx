/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, DialogTrigger } from '@geti/ui';

import { AddLabelDialog } from './add-label-dialog.component';

import classes from './add-label.module.scss';

interface AddLabelProps {
    existingLabelsNames: string[];
}

export const AddLabel = ({ existingLabelsNames }: AddLabelProps) => {
    return (
        <DialogTrigger type={'popover'} hideArrow placement={'bottom right'}>
            <Button variant={'secondary'} UNSAFE_className={classes.addLabelButton}>
                Add label
            </Button>
            {(close) => <AddLabelDialog closeDialog={close} existingLabelsNames={existingLabelsNames} />}
        </DialogTrigger>
    );
};
