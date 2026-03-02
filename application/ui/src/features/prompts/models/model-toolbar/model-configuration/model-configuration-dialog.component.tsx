/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { FormEvent, useState } from 'react';

import { MatcherModel, ModelType, PerDINOModel, SoftMatcherModel } from '@/api';
import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Form, Heading, Item, Picker, Switch } from '@geti/ui';

import { useUpdateModel } from '../../api/use-update-model';
import { NumberField } from './number-field.component';
import { isMatcherModel, isPerDINOModel, isSoftMatcherModel } from './utils';

const ENCODER_MODELS = [
    { label: 'DINOv3 Small', value: 'dinov3_small' },
    { label: 'DINOv3 Small Plus', value: 'dinov3_small_plus' },
    { label: 'DINOv3 Base', value: 'dinov3_base' },
    { label: 'DINOv3 Large', value: 'dinov3_large' },
    { label: 'DINOv3 Huge', value: 'dinov3_huge' },
] as const;

// ATM backend does not provide a list of available models, so we have to hardcode them here.
type EncoderModel = (typeof ENCODER_MODELS)[number]['value'];

type DecoderModel = ModelType['config']['sam_model'];

const DECODER_MODELS: { label: string; value: DecoderModel }[] = [
    {
        label: 'SAM-HQ',
        value: 'SAM-HQ',
    },
    {
        label: 'SAM-HQ Tiny',
        value: 'SAM-HQ-tiny',
    },
    {
        label: 'SAM2 Tiny',
        value: 'SAM2-tiny',
    },
    {
        label: 'SAM2 Small',
        value: 'SAM2-small',
    },
    {
        label: 'SAM2 Base',
        value: 'SAM2-base',
    },
    {
        label: 'SAM2 Large',
        value: 'SAM2-large',
    },
];

type Precision = 'fp16' | 'fp32' | 'bf16';

const PRECISIONS: { label: string; value: Precision }[] = [
    { label: 'FP16', value: 'fp16' },
    { label: 'FP32', value: 'fp32' },
    { label: 'BF16', value: 'bf16' },
];

interface SelectionProps<T extends string> {
    value: T;
    onChange: (model: T) => void;
    label: string;
    items: Iterable<{ label: string; value: T }>;
}

const Selection = <T extends string>({ value, onChange, label, items }: SelectionProps<T>) => {
    return (
        <Picker label={label} items={items} onSelectionChange={(v) => onChange(v as T)} selectedKey={value}>
            {(item) => <Item key={item.value}>{item.label}</Item>}
        </Picker>
    );
};

interface MatcherConfigurationProps {
    model: MatcherModel;
    onClose: () => void;
}

const MatcherConfiguration = ({ model, onClose }: MatcherConfigurationProps) => {
    const [numberOfForegroundPoints, setNumberOfForegroundPoints] = useState<number>(
        model.config.num_foreground_points
    );
    const [numberOfBackgroundPoints, setNumberOfBackgroundPoints] = useState<number>(
        model.config.num_background_points
    );
    const [confidenceThreshold, setConfidenceThreshold] = useState<number>(model.config.confidence_threshold);
    const [encoderModel, setEncoderModel] = useState<EncoderModel>(model.config.encoder_model as EncoderModel);
    const [decoderModel, setDecoderModel] = useState<DecoderModel>(model.config.sam_model);
    const [precision, setPrecision] = useState<Precision>(model.config.precision as Precision);
    const [useMaskRefinement, setUseMaskRefinement] = useState<boolean>(model.config.use_mask_refinement);
    const [useNMS, setUseNMS] = useState<boolean>(model.config.use_nms);

    const updateModelMutation = useUpdateModel();

    const isConfigureButtonDisabled =
        numberOfForegroundPoints === model.config.num_foreground_points &&
        numberOfBackgroundPoints === model.config.num_background_points &&
        confidenceThreshold === model.config.confidence_threshold &&
        encoderModel === model.config.encoder_model &&
        decoderModel === model.config.sam_model &&
        precision === model.config.precision &&
        useMaskRefinement === model.config.use_mask_refinement &&
        useNMS === model.config.use_nms;

    const updateModel = (event: FormEvent) => {
        event.preventDefault();

        updateModelMutation.mutate(
            {
                active: model.active,
                name: model.name,
                id: model.id,
                config: {
                    model_type: model.config.model_type,
                    num_foreground_points: numberOfForegroundPoints,
                    num_background_points: numberOfBackgroundPoints,
                    confidence_threshold: confidenceThreshold,
                    encoder_model: encoderModel,
                    sam_model: decoderModel,
                    use_mask_refinement: useMaskRefinement,
                    use_nms: useNMS,
                    precision,
                },
            },
            onClose
        );
    };

    return (
        <Form onSubmit={updateModel}>
            <Flex direction={'column'} gap={'size-200'}>
                <Flex alignItems={'center'} gap={'size-200'}>
                    <Selection
                        label={'Encoder model'}
                        items={ENCODER_MODELS}
                        value={encoderModel}
                        onChange={setEncoderModel}
                    />
                    <Selection
                        label={'Decoder model'}
                        items={DECODER_MODELS}
                        value={decoderModel}
                        onChange={setDecoderModel}
                    />
                </Flex>
                <NumberField
                    label={'Number of foreground points'}
                    minValue={1}
                    maxValue={300}
                    step={1}
                    onChange={setNumberOfForegroundPoints}
                    value={numberOfForegroundPoints}
                />
                <NumberField
                    label={'Number of background points'}
                    minValue={0}
                    maxValue={10}
                    step={1}
                    onChange={setNumberOfBackgroundPoints}
                    value={numberOfBackgroundPoints}
                />
                <NumberField
                    label={'Confidence threshold'}
                    minValue={0}
                    maxValue={1}
                    step={0.01}
                    onChange={setConfidenceThreshold}
                    value={confidenceThreshold}
                />
                <Selection label={'Precision'} value={precision} onChange={setPrecision} items={PRECISIONS} />
                <Flex alignItems={'center'} width={'100%'} wrap={'wrap'}>
                    <Switch isEmphasized isSelected={useMaskRefinement} onChange={setUseMaskRefinement}>
                        Use mask refinement
                    </Switch>
                    <Switch isEmphasized isSelected={useNMS} onChange={setUseNMS}>
                        Merge overlapping results
                    </Switch>
                </Flex>
                <ButtonGroup align={'end'}>
                    <Button variant={'secondary'} onPress={onClose}>
                        Cancel
                    </Button>
                    <Button
                        type={'submit'}
                        variant={'primary'}
                        isPending={updateModelMutation.isPending}
                        isDisabled={isConfigureButtonDisabled}
                    >
                        Configure
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};

interface PerDINOConfigurationProps {
    model: PerDINOModel;
    onClose: () => void;
}

const PerDINOConfiguration = ({ model, onClose }: PerDINOConfigurationProps) => {
    const [numberOfForegroundPoints, setNumberOfForegroundPoints] = useState<number>(
        model.config.num_foreground_points
    );
    const [numberOfBackgroundPoints, setNumberOfBackgroundPoints] = useState<number>(
        model.config.num_background_points
    );
    const [numberOfGridCells, setNumberOfNumberOfGridCells] = useState<number>(model.config.num_grid_cells);
    const [confidenceThreshold, setConfidenceThreshold] = useState<number>(model.config.confidence_threshold);
    const [encoderModel, setEncoderModel] = useState<EncoderModel>(model.config.encoder_model as EncoderModel);
    const [decoderModel, setDecoderModel] = useState<DecoderModel>(model.config.sam_model);
    const [precision, setPrecision] = useState<Precision>(model.config.precision as Precision);
    const [useNMS, setUseNMS] = useState<boolean>(model.config.use_nms);
    const [pointSelectionThreshold, setPointSelectionThreshold] = useState<number>(
        model.config.point_selection_threshold
    );

    const updateModelMutation = useUpdateModel();

    const isConfigureButtonDisabled =
        numberOfForegroundPoints === model.config.num_foreground_points &&
        numberOfBackgroundPoints === model.config.num_background_points &&
        numberOfGridCells === model.config.num_grid_cells &&
        confidenceThreshold === model.config.confidence_threshold &&
        pointSelectionThreshold === model.config.point_selection_threshold &&
        encoderModel === model.config.encoder_model &&
        decoderModel === model.config.sam_model &&
        precision === model.config.precision &&
        useNMS === model.config.use_nms;

    const updateModel = (event: FormEvent) => {
        event.preventDefault();

        updateModelMutation.mutate(
            {
                active: model.active,
                name: model.name,
                id: model.id,
                config: {
                    model_type: model.config.model_type,
                    num_foreground_points: numberOfForegroundPoints,
                    num_background_points: numberOfBackgroundPoints,
                    num_grid_cells: numberOfGridCells,
                    confidence_threshold: confidenceThreshold,
                    point_selection_threshold: pointSelectionThreshold,
                    encoder_model: encoderModel,
                    sam_model: decoderModel,
                    use_nms: useNMS,
                    precision,
                },
            },
            onClose
        );
    };

    return (
        <Form onSubmit={updateModel}>
            <Flex direction={'column'} gap={'size-200'}>
                <Flex alignItems={'center'} gap={'size-200'}>
                    <Selection
                        label={'Encoder model'}
                        items={ENCODER_MODELS}
                        value={encoderModel}
                        onChange={setEncoderModel}
                    />
                    <Selection
                        label={'Decoder model'}
                        items={DECODER_MODELS}
                        value={decoderModel}
                        onChange={setDecoderModel}
                    />
                </Flex>
                <NumberField
                    label={'Number of foreground points'}
                    minValue={1}
                    maxValue={300}
                    step={1}
                    onChange={setNumberOfForegroundPoints}
                    value={numberOfForegroundPoints}
                />
                <NumberField
                    label={'Number of background points'}
                    minValue={0}
                    maxValue={10}
                    step={1}
                    onChange={setNumberOfBackgroundPoints}
                    value={numberOfBackgroundPoints}
                />
                <NumberField
                    label={'Number of grid cells'}
                    minValue={1}
                    maxValue={100}
                    step={1}
                    onChange={setNumberOfNumberOfGridCells}
                    value={numberOfGridCells}
                />
                <NumberField
                    label={'Confidence threshold'}
                    minValue={0}
                    maxValue={1}
                    step={0.01}
                    onChange={setConfidenceThreshold}
                    value={confidenceThreshold}
                />
                <NumberField
                    label={'Point selection threshold'}
                    minValue={0}
                    maxValue={1}
                    step={0.01}
                    onChange={setPointSelectionThreshold}
                    value={pointSelectionThreshold}
                />
                <Selection label={'Precision'} value={precision} onChange={setPrecision} items={PRECISIONS} />
                <Flex alignItems={'center'} width={'100%'} wrap={'wrap'}>
                    <Switch isEmphasized isSelected={useNMS} onChange={setUseNMS}>
                        Merge overlapping results
                    </Switch>
                </Flex>
                <ButtonGroup align={'end'}>
                    <Button variant={'secondary'} onPress={onClose}>
                        Cancel
                    </Button>
                    <Button
                        type={'submit'}
                        variant={'primary'}
                        isPending={updateModelMutation.isPending}
                        isDisabled={isConfigureButtonDisabled}
                    >
                        Configure
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};

interface SoftMatcherConfigurationProps {
    model: SoftMatcherModel;
    onClose: () => void;
}

const SoftMatcherConfiguration = ({ model, onClose }: SoftMatcherConfigurationProps) => {
    const [numberOfForegroundPoints, setNumberOfForegroundPoints] = useState<number>(
        model.config.num_foreground_points
    );
    const [numberOfBackgroundPoints, setNumberOfBackgroundPoints] = useState<number>(
        model.config.num_background_points
    );
    const [confidenceThreshold, setConfidenceThreshold] = useState<number>(model.config.confidence_threshold);
    const [encoderModel, setEncoderModel] = useState<EncoderModel>(model.config.encoder_model as EncoderModel);
    const [decoderModel, setDecoderModel] = useState<DecoderModel>(model.config.sam_model);
    const [precision, setPrecision] = useState<Precision>(model.config.precision as Precision);
    const [useNMS, setUseNMS] = useState<boolean>(model.config.use_nms);
    const [useSampling, setUseSampling] = useState<boolean>(model.config.use_sampling);
    const [useSpatialSampling, setUseSpatialSampling] = useState<boolean>(model.config.use_spatial_sampling);
    const [approximateMatching, setApproximateMatching] = useState<boolean>(model.config.approximate_matching);
    const [softMatchingScoreThreshold, setSoftMatchingScoreThreshold] = useState<number>(
        model.config.softmatching_score_threshold
    );
    const [softMatchingBidirectional, setSoftMatchingBidirectional] = useState<boolean>(
        model.config.softmatching_bidirectional
    );

    const updateModelMutation = useUpdateModel();

    const isConfigureButtonDisabled =
        numberOfForegroundPoints === model.config.num_foreground_points &&
        numberOfBackgroundPoints === model.config.num_background_points &&
        confidenceThreshold === model.config.confidence_threshold &&
        encoderModel === model.config.encoder_model &&
        decoderModel === model.config.sam_model &&
        precision === model.config.precision &&
        useNMS === model.config.use_nms &&
        useSampling === model.config.use_sampling &&
        useSpatialSampling === model.config.use_spatial_sampling &&
        approximateMatching === model.config.approximate_matching &&
        softMatchingScoreThreshold === model.config.softmatching_score_threshold &&
        softMatchingBidirectional === model.config.softmatching_bidirectional;

    const updateModel = (event: FormEvent) => {
        event.preventDefault();

        updateModelMutation.mutate(
            {
                active: model.active,
                name: model.name,
                id: model.id,
                config: {
                    model_type: model.config.model_type,
                    num_foreground_points: numberOfForegroundPoints,
                    num_background_points: numberOfBackgroundPoints,
                    confidence_threshold: confidenceThreshold,
                    encoder_model: encoderModel,
                    sam_model: decoderModel,
                    use_nms: useNMS,
                    softmatching_bidirectional: softMatchingBidirectional,
                    softmatching_score_threshold: softMatchingScoreThreshold,
                    approximate_matching: approximateMatching,
                    use_sampling: useSampling,
                    use_spatial_sampling: useSpatialSampling,
                    precision,
                },
            },
            onClose
        );
    };

    return (
        <Form onSubmit={updateModel}>
            <Flex direction={'column'} gap={'size-200'}>
                <Flex alignItems={'center'} gap={'size-200'}>
                    <Selection
                        label={'Encoder model'}
                        items={ENCODER_MODELS}
                        value={encoderModel}
                        onChange={setEncoderModel}
                    />
                    <Selection
                        label={'Decoder model'}
                        items={DECODER_MODELS}
                        value={decoderModel}
                        onChange={setDecoderModel}
                    />
                </Flex>
                <NumberField
                    label={'Number of foreground points'}
                    minValue={1}
                    maxValue={300}
                    step={1}
                    onChange={setNumberOfForegroundPoints}
                    value={numberOfForegroundPoints}
                />
                <NumberField
                    label={'Number of background points'}
                    minValue={0}
                    maxValue={10}
                    step={1}
                    onChange={setNumberOfBackgroundPoints}
                    value={numberOfBackgroundPoints}
                />
                <NumberField
                    label={'Confidence threshold'}
                    minValue={0}
                    maxValue={1}
                    step={0.01}
                    onChange={setConfidenceThreshold}
                    value={confidenceThreshold}
                />
                <NumberField
                    label={'Soft matching score threshold'}
                    minValue={0}
                    maxValue={1}
                    step={0.01}
                    onChange={setSoftMatchingScoreThreshold}
                    value={softMatchingScoreThreshold}
                />
                <Selection label={'Precision'} value={precision} onChange={setPrecision} items={PRECISIONS} />
                <Flex alignItems={'center'} width={'100%'} wrap={'wrap'}>
                    <Switch isEmphasized isSelected={softMatchingBidirectional} onChange={setSoftMatchingBidirectional}>
                        Bidirectional soft matching
                    </Switch>
                    <Switch isEmphasized isSelected={useNMS} onChange={setUseNMS}>
                        Merge overlapping results
                    </Switch>
                    <Switch isEmphasized isSelected={approximateMatching} onChange={setApproximateMatching}>
                        Approximate matching
                    </Switch>
                    <Switch isEmphasized isSelected={useSpatialSampling} onChange={setUseSpatialSampling}>
                        Use spatial sampling
                    </Switch>
                    <Switch isEmphasized isSelected={useSampling} onChange={setUseSampling}>
                        Use sampling
                    </Switch>
                </Flex>
                <ButtonGroup align={'end'}>
                    <Button variant={'secondary'} onPress={onClose}>
                        Cancel
                    </Button>
                    <Button
                        type={'submit'}
                        variant={'primary'}
                        isPending={updateModelMutation.isPending}
                        isDisabled={isConfigureButtonDisabled}
                    >
                        Configure
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};

interface ModelConfigurationDialogProps {
    model: ModelType;
    onClose: () => void;
}

export const ModelConfigurationDialog = ({ model, onClose }: ModelConfigurationDialogProps) => {
    return (
        <Dialog width={'40vw'}>
            <Heading>Model configuration</Heading>
            <Divider size={'S'} />
            <Content>
                {isMatcherModel(model) ? (
                    <MatcherConfiguration model={model} onClose={onClose} />
                ) : isPerDINOModel(model) ? (
                    <PerDINOConfiguration model={model} onClose={onClose} />
                ) : isSoftMatcherModel(model) ? (
                    <SoftMatcherConfiguration model={model} onClose={onClose} />
                ) : null}
            </Content>
        </Dialog>
    );
};
