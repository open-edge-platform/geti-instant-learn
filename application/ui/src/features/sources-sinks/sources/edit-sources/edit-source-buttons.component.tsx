/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, ButtonGroup } from '@geti/ui';

interface EditSourceButtonsProps {
    isActiveSource: boolean;
    onSave: () => void;
    onSaveAndConnect: () => void;
    isDisabled: boolean;
    isPending: boolean;
}

export const EditSourceButtons = ({
    isActiveSource,
    onSave,
    onSaveAndConnect,
    isDisabled,
    isPending,
}: EditSourceButtonsProps) => {
    return (
        <ButtonGroup>
            {isActiveSource ? (
                <Button type={'submit'} onPress={onSave} isPending={isPending} isDisabled={isDisabled}>
                    Save
                </Button>
            ) : (
                <>
                    <Button type={'submit'} onPress={onSave} isPending={isPending} isDisabled={isDisabled}>
                        Save
                    </Button>
                    <Button onPress={onSaveAndConnect} isPending={isPending} isDisabled={isDisabled}>
                        Save & Connect
                    </Button>
                </>
            )}
        </ButtonGroup>
    );
};
