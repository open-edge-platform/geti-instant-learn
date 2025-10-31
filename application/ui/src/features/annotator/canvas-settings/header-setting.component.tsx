/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionButton, Flex, Text, Tooltip, TooltipTrigger } from '@geti/ui';
import { Revisit } from '@geti/ui/icons';
import { useNumberFormatter } from 'react-aria';

export interface HeaderSettingProps {
    headerText: string;
    value: number;
    defaultValue: number;
    handleValueChange: (value: number) => void;
    formatOptions: Intl.NumberFormatOptions;
}

export const HeaderSetting = ({
    headerText,
    value,
    handleValueChange,
    formatOptions,
    defaultValue,
}: HeaderSettingProps) => {
    const formatter = useNumberFormatter(formatOptions);

    return (
        <div>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Text>{headerText}</Text>
                <Flex alignItems={'center'} gap={'size-100'} height={'size-400'}>
                    <TooltipTrigger placement={'top'}>
                        <ActionButton
                            isQuiet
                            onPress={() => handleValueChange(defaultValue)}
                            aria-label={`Reset ${headerText.toLocaleLowerCase()}`}
                        >
                            <Revisit />
                        </ActionButton>
                        <Tooltip>{`Reset to default value ${headerText.toLocaleLowerCase()}`}</Tooltip>
                    </TooltipTrigger>

                    <Text>{formatter.format(value)}</Text>
                </Flex>
            </Flex>
        </div>
    );
};
