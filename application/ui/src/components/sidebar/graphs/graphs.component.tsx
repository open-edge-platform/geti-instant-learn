/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, Flex, Heading, IllustratedMessage, View } from '@geti/ui';
import { CartesianGrid, Label, Line, LineChart, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from 'recharts';

interface DataPoint {
    name: string;
    value: number;
}

const useMetricsData = () => {
    // TODO: Replace with real API call when endpoint is implemented
    const generateMockData = (baseValue: number, variance: number, count: number = 30): DataPoint[] => {
        return Array.from({ length: count }, (_, i) => ({
            name: `${i}s`,
            value: baseValue + (Math.random() - 0.5) * variance,
        }));
    };

    const latencyData = generateMockData(45, 15); // ~30-60ms latency
    const throughputData = generateMockData(25, 8); // ~21-29 requests/sec
    const metrics = {
        avgLatency: 45,
        avgThroughput: 25,
    };

    return { latencyData, throughputData, metrics };
};

const Graph = ({ label, data }: { label: string; data: DataPoint[] }) => {
    return (
        <View padding={{ top: 'size-250', left: 'size-200', right: 'size-200', bottom: 'size-125' }}>
            <ResponsiveContainer width='100%' height={200}>
                <LineChart data={data}>
                    <XAxis
                        minTickGap={32}
                        stroke='var(--spectrum-global-color-gray-800)'
                        dataKey='name'
                        tickLine={false}
                        tickMargin={8}
                    />
                    <YAxis
                        tickLine={false}
                        stroke='var(--spectrum-global-color-gray-900)'
                        dataKey='value'
                        tickFormatter={(value: number) => {
                            return value > 10 ? value.toFixed(0) : value.toFixed(2);
                        }}
                    >
                        <Label
                            angle={-90}
                            value={label}
                            position='insideLeft'
                            offset={10}
                            style={{
                                textAnchor: 'middle',
                                fill: 'var(--spectrum-global-color-gray-900)',
                                fontSize: '10px',
                            }}
                        />
                    </YAxis>
                    <CartesianGrid stroke='var(--spectrum-global-color-gray-400)' />
                    {data.length > 0 && (
                        <ReferenceLine
                            x={data[0].name}
                            stroke='var(--spectrum-global-color-gray-600)'
                            strokeWidth={2}
                        />
                    )}
                    <Line
                        type='linear'
                        dataKey='value'
                        dot={false}
                        stroke='var(--energy-blue)'
                        isAnimationActive={false}
                        strokeWidth='3'
                    />
                </LineChart>
            </ResponsiveContainer>
        </View>
    );
};

export const Graphs = () => {
    const { latencyData, throughputData, metrics } = useMetricsData();

    const hasData = latencyData.length > 0 || throughputData.length > 0;

    return (
        <View
            minWidth={'size-4600'}
            width={'100%'}
            backgroundColor={'gray-100'}
            paddingY={'size-200'}
            paddingX={'size-300'}
            height={'100%'}
        >
            <Flex direction={'column'} height={'100%'}>
                <Heading margin={0}>Model statistics</Heading>
                <View padding={'size-300'} flex={1} UNSAFE_style={{ overflow: 'hidden auto' }}>
                    {!hasData && !metrics ? (
                        <IllustratedMessage>
                            <Heading>No statistics available</Heading>
                            <Content>
                                Model statistics will appear here once the pipeline starts running and starts processing
                                data.
                            </Content>
                        </IllustratedMessage>
                    ) : (
                        <Flex direction={'column'} gap={'size-300'} height={'100%'}>
                            <View>
                                <Heading level={4} marginBottom={'size-300'}>
                                    Throughput
                                </Heading>
                                <Graph label='requests/sec' data={throughputData} />
                            </View>
                            <View>
                                <Heading level={4} marginBottom={'size-300'}>
                                    Latency
                                </Heading>
                                <Graph label='ms' data={latencyData} />
                            </View>
                        </Flex>
                    )}
                </View>
            </Flex>
        </View>
    );
};
