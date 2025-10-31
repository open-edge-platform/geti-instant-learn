/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject } from 'react';

import {
    ActionButton,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    DOMRefValue,
    Flex,
    Heading,
    Text,
    useUnwrapDOMRef,
} from '@geti/ui';
import { Adjustments, Close } from '@geti/ui/icons';

import { useFullScreenMode } from '../../prompts/visual-prompt/captured-frame/full-screen-mode.component';
import { useCanvasSettings } from './canvas-settings-provider.component';
import { SettingsList } from './settings-list.component';

import classes from './canvas-settings.module.scss';

interface CanvasSettingsProps {
    ref: RefObject<DOMRefValue<HTMLDivElement> | null>;
}

export const CanvasSettings = ({ ref }: CanvasSettingsProps) => {
    const { canvasSettings, setCanvasSettings } = useCanvasSettings();
    const targetRef = useUnwrapDOMRef(ref);
    const { isFullScreenMode } = useFullScreenMode();

    return (
        <DialogTrigger
            type={'popover'}
            hideArrow
            targetRef={targetRef}
            placement={isFullScreenMode ? 'top right' : 'right'}
        >
            <ActionButton isQuiet aria-label={'Canvas settings'}>
                <Adjustments />
            </ActionButton>
            {(close) => (
                <Dialog UNSAFE_className={classes.canvasDialog}>
                    <Heading>
                        <Flex justifyContent={'space-between'} alignItems={'center'}>
                            <Text>Canvas settings</Text>
                            <ActionButton isQuiet onPress={close} aria-label={'Close canvas settings'}>
                                <Close />
                            </ActionButton>
                        </Flex>
                    </Heading>
                    <Divider marginY={'size-150'} UNSAFE_className={classes.canvasAdjustmentsDivider} />
                    <Content>
                        <SettingsList canvasSettings={canvasSettings} onCanvasSettingsChange={setCanvasSettings} />
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
