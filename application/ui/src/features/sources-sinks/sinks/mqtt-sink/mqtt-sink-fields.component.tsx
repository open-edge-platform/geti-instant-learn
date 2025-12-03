/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Flex, NumberField, Switch, TextField } from '@geti/ui';

export const MQTTSinkFields = () => {
    return (
        <>
            <TextField isRequired label={'Name'} name={'name'} />
            <TextField isRequired label={'Broker Host'} name={'broker_host'} />
            <Flex alignItems={'center'} gap={'size-200'}>
                <TextField isRequired flex={1} label={'Topic'} name={'topic'} />
                <NumberField isRequired flexBasis={'40%'} label={'Broker port'} name={'broker_port'} />
            </Flex>

            <Switch isEmphasized name={'auth_required'}>
                Auth required
            </Switch>
        </>
    );
};
