/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Checkbox, CheckboxGroup, Flex, NumberField, TextField } from '@geti/ui';

export const MQTTSinkFields = () => {
    return (
        <>
            <TextField label={'Name'} />
            <TextField label={'Broker Host'} />
            <Flex alignItems={'center'} gap={'size-200'}>
                <TextField flex={1} label={'Topic'} />
                <NumberField flexBasis={'40%'} label={'Broker port'} />
            </Flex>
            <NumberField width={'40%'} label={'Rate limit'} />
            <CheckboxGroup label={'Output formats'}>
                <Checkbox isEmphasized>Predictions</Checkbox>
                <Checkbox isEmphasized>Image original</Checkbox>
                <Checkbox isEmphasized>Image with predictions</Checkbox>
            </CheckboxGroup>
        </>
    );
};
