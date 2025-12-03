/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { ActionButton, Divider, Flex, Heading, View } from '@geti/ui';
import { Back } from '@geti/ui/icons';

interface PipelineEntityTitleProps {
    children: ReactNode;
}

const PipelineEntityTitle = ({ children }: PipelineEntityTitleProps) => {
    return (
        <Heading margin={0} UNSAFE_style={{ fontWeight: 500 }}>
            {children}
        </Heading>
    );
};

interface PipelineEntityDescriptionProps {
    children: ReactNode;
    title: string;
}

const PipelineEntityContent = ({ children, title }: PipelineEntityDescriptionProps) => {
    return (
        <View
            padding={'size-200'}
            backgroundColor={'gray-100'}
            borderColor={'gray-200'}
            borderWidth={'thin'}
            borderRadius={'regular'}
        >
            <Heading margin={0} marginBottom={'size-200'}>
                {title}
            </Heading>
            {children}
        </View>
    );
};

interface PipelineEntityPanel {
    title: ReactNode;
    onBackClick: () => void;
    children: ReactNode;
}

export const PipelineEntityPanel = ({ title, children, onBackClick }: PipelineEntityPanel) => {
    return (
        <View>
            <Flex alignItems={'center'} gap={'size-75'}>
                <ActionButton isQuiet aria-label={'Go back'} onPress={onBackClick}>
                    <Back />
                </ActionButton>
                {title}
            </Flex>
            <Divider size={'S'} marginY={'size-200'} />
            {children}
        </View>
    );
};

PipelineEntityPanel.Title = PipelineEntityTitle;
PipelineEntityPanel.Content = PipelineEntityContent;
