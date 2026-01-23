/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, NumberField as RootNumberField, Slider } from '@geti/ui';

interface NumberFieldProps {
    label: string;
    minValue: number;
    maxValue: number;
    step: number;
    onChange: (value: number) => void;
    value: number;
}

export const NumberField = ({ label, maxValue, value, minValue, onChange, step }: NumberFieldProps) => {
    return (
        <Flex alignItems={'end'} gap={'size-100'} width={'100%'}>
            <RootNumberField
                label={label}
                value={value}
                onChange={onChange}
                step={step}
                minValue={minValue}
                maxValue={maxValue}
            />
            <Slider
                flex={1}
                isFilled
                aria-label={label}
                value={value}
                onChange={onChange}
                step={step}
                minValue={minValue}
                maxValue={maxValue}
            />
        </Flex>
    );
};
