/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button } from '@geti/ui';
import { Add as AddIcon } from '@geti/ui/icons';

import styles from './add-new-source.module.scss';

interface AddNewSourceProps {
    onAddNewSource: () => void;
}

export const AddNewSource = ({ onAddNewSource }: AddNewSourceProps) => {
    return (
        <Button variant={'secondary'} onPress={onAddNewSource} UNSAFE_className={styles.addNewSourceButton}>
            <AddIcon /> Add new source
        </Button>
    );
};
