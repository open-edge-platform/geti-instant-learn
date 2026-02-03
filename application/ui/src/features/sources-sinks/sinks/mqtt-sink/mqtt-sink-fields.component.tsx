/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Dispatch, SetStateAction } from 'react';

import { MQTTSinkType } from '@/api';
import { Flex, NumberField, Switch, TextField } from '@geti/ui';

interface MQTTSinkFieldsProps {
    sinkConfig: MQTTSinkType['config'];
    onSetSinkConfig: Dispatch<SetStateAction<MQTTSinkType['config']>>;
}

export const MQTTSinkFields = ({ sinkConfig, onSetSinkConfig }: MQTTSinkFieldsProps) => {
    return (
        <>
            <TextField
                isRequired
                label={'Name'}
                name={'name'}
                value={sinkConfig.name}
                onChange={(name) => onSetSinkConfig((prevSinkConfig) => ({ ...prevSinkConfig, name }))}
            />
            <TextField
                isRequired
                label={'Broker host'}
                name={'broker_host'}
                value={sinkConfig.broker_host}
                onChange={(brokerHost) =>
                    onSetSinkConfig((prevSinkConfig) => ({ ...prevSinkConfig, broker_host: brokerHost }))
                }
            />
            <Flex alignItems={'center'} gap={'size-200'}>
                <TextField
                    isRequired
                    flex={1}
                    label={'Topic'}
                    name={'topic'}
                    value={sinkConfig.topic}
                    onChange={(topic) => onSetSinkConfig((prevSinkConfig) => ({ ...prevSinkConfig, topic }))}
                />
                <NumberField
                    isRequired
                    flexBasis={'40%'}
                    label={'Broker port'}
                    name={'broker_port'}
                    value={sinkConfig.broker_port}
                    onChange={(brokerPort) =>
                        onSetSinkConfig((prevSinkConfig) => ({ ...prevSinkConfig, broker_port: brokerPort }))
                    }
                />
            </Flex>

            <Switch
                isEmphasized
                name={'auth_required'}
                isSelected={sinkConfig.auth_required}
                onChange={(authRequired) =>
                    onSetSinkConfig((prevSinkConfig) => ({ ...prevSinkConfig, auth_required: authRequired }))
                }
            >
                Auth required
            </Switch>
        </>
    );
};
