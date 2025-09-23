/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { use, useState } from 'react';

import { Button, Content, Dialog, DialogTrigger, Flex } from '@geti/ui';

import { ChangeColorButton } from './change-color-button.component';
import { ColorPickerDialog } from './color-picker-dialog.component';

export const Labels = () => {
    const [color, setColor] = useState<string>('#ededed');
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'end'}>
            <DialogTrigger type={'popover'} hideArrow placement={'bottom right'}>
                <Button variant={'secondary'} UNSAFE_style={{ border: 'none' }}>
                    Add label
                </Button>
                {(_close) => (
                    <Dialog>
                        <Content>
                            <ColorPickerDialog
                                color={color}
                                id={'change-color-button'}
                                data-testid={'change-color-button'}
                                onColorChange={setColor}
                                size={'M'}
                                gridArea={'color'}
                                ariaLabelPrefix={'tmp'}
                            />
                        </Content>
                    </Dialog>
                )}
            </DialogTrigger>
        </Flex>
    );
};
