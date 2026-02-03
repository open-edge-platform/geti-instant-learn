/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { RefObject } from 'react';

import { LabelType } from '@/api';
import { Point } from '@geti/smart-tools/types';
import { ActionButton, Flex, View } from '@geti/ui';
import { Close } from '@geti/ui/icons';
import { ThemeProvider } from '@geti/ui/theme';
import { Popover } from 'react-aria-components';

import { CreateLabelForm } from '../../../prompts/visual-prompt/labels-management/add-label/create-label-form.component';

interface CreateLabelPopoverProps {
    onSuccess: (label: LabelType) => void;
    existingLabels: LabelType[];
    mousePosition: Point | null;
    onClose: () => void;
    ref: RefObject<SVGSVGElement | null>;
}

export const CreateLabelPopover = ({
    onSuccess,
    existingLabels,
    onClose,
    mousePosition,
    ref,
}: CreateLabelPopoverProps) => {
    if (mousePosition === null) return null;

    return (
        <Popover
            isOpen
            offset={mousePosition.y}
            crossOffset={mousePosition.x}
            placement={'bottom start'}
            onOpenChange={onClose}
            triggerRef={ref}
            style={{
                transform: 'translate(-50%, -50%)',
            }}
        >
            <ThemeProvider>
                <View
                    backgroundColor={'gray-50'}
                    padding={'size-200'}
                    height={'100%'}
                    borderRadius={'regular'}
                    borderWidth={'thin'}
                    borderColor={'gray-400'}
                >
                    <Flex justifyContent={'space-between'} gap={'size-100'}>
                        <CreateLabelForm onClose={onClose} onSuccess={onSuccess} existingLabels={existingLabels} />
                        <ActionButton isQuiet onPress={onClose}>
                            <Close />
                        </ActionButton>
                    </Flex>
                </View>
            </ThemeProvider>
        </Popover>
    );
};
