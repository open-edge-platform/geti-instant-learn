/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, Content, Dialog, DialogTrigger, Flex } from '@geti/ui';

import { ChangeColorButton } from './change-color-button.component';

export const Labels = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'end'}>
            <DialogTrigger type={'popover'} hideArrow placement={'bottom right'}>
                <Button variant={'secondary'} UNSAFE_style={{ border: 'none' }}>
                    Add label
                </Button>
                {(_close) => (
                    <Dialog>
                        <Content>
                            <ChangeColorButton color='#ededed' id='label-color-button' size='M' />
                        </Content>
                    </Dialog>
                )}
            </DialogTrigger>
        </Flex>
    );
};
