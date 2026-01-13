/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelType, ModelUpdateType } from '@geti-prompt/api';
import { getMockedModel, render } from '@geti-prompt/test-utils';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http, server } from '../../../../../setup-test';
import { ModelConfigurationDialog } from './model-configuration-dialog.component';

class ModelConfigurationDialogPage {
    constructor() {}

    async closeDialog() {
        await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    }

    get configureButton() {
        return screen.getByRole('button', { name: 'Configure' });
    }

    private async changeNumberField(field: string, value: number) {
        const numberField = screen.getByRole('textbox', { name: field });
        await userEvent.clear(numberField);
        await userEvent.type(numberField, value.toString());
        fireEvent.blur(numberField);
    }

    async changeNumberOfForegroundPointes(value: number) {
        await this.changeNumberField('Number of foreground points', value);
    }

    async changeNumberOfBackgroundPointes(value: number) {
        await this.changeNumberField('Number of background points', value);
    }

    async changeConfidenceThreshold(value: number) {
        await this.changeNumberField('Confidence threshold', value);
    }

    private async changeSelection(field: string, value: string) {
        await userEvent.click(screen.getByRole('button', { name: new RegExp(field) }));
        await userEvent.click(screen.getByRole('option', { name: value }));
    }

    async changeEncoderModel(value: string) {
        await this.changeSelection('Encoder model', value);
    }

    async changeDecoderModel(value: string) {
        await this.changeSelection('Decoder model', value);
    }

    async changePrecision(value: string) {
        await this.changeSelection('Precision', value);
    }

    async changeUseMaskRefinement() {
        await userEvent.click(screen.getByRole('switch', { name: 'Use mask refinement' }));
    }

    async configureModel() {
        await userEvent.click(this.configureButton);
    }
}

const renderModelConfigurationDialog = ({
    model = getMockedModel(),
    onClose = vi.fn(),
}: { model?: ModelType; onClose?: () => void } = {}) => {
    const result = render(<ModelConfigurationDialog model={model} onClose={onClose} />);

    return {
        result,
        modelConfigurationDialogPage: new ModelConfigurationDialogPage(),
    };
};

describe('ModelConfigurationDialog', () => {
    it('disables configure button when parameters have not been changed', () => {
        const model = getMockedModel();
        const { modelConfigurationDialogPage } = renderModelConfigurationDialog({ model });

        expect(modelConfigurationDialogPage.configureButton).toBeDisabled();
    });

    it('enables configure button when any of the parameters has been changes', async () => {
        const mockedModel = getMockedModel();

        const model = getMockedModel({
            config: {
                ...mockedModel.config,
                num_foreground_points: 10,
                num_background_points: 10,
                confidence_threshold: 0.2,
                sam_model: 'SAM-HQ-tiny',
                encoder_model: 'dinov3_small',
                use_mask_refinement: true,
                precision: 'bf16',
            },
        });

        const { modelConfigurationDialogPage } = renderModelConfigurationDialog({ model });

        await modelConfigurationDialogPage.changeNumberOfForegroundPointes(20);
        expect(modelConfigurationDialogPage.configureButton).toBeEnabled();
        await modelConfigurationDialogPage.changeNumberOfForegroundPointes(model.config.num_foreground_points);
        expect(modelConfigurationDialogPage.configureButton).toBeDisabled();

        await modelConfigurationDialogPage.changeNumberOfBackgroundPointes(50);
        expect(modelConfigurationDialogPage.configureButton).toBeEnabled();
        await modelConfigurationDialogPage.changeNumberOfBackgroundPointes(model.config.num_background_points);
        expect(modelConfigurationDialogPage.configureButton).toBeDisabled();

        await modelConfigurationDialogPage.changeConfidenceThreshold(0.8);
        expect(modelConfigurationDialogPage.configureButton).toBeEnabled();
        await modelConfigurationDialogPage.changeConfidenceThreshold(model.config.confidence_threshold);
        expect(modelConfigurationDialogPage.configureButton).toBeDisabled();

        await modelConfigurationDialogPage.changeEncoderModel('DINOv3 Base');
        expect(modelConfigurationDialogPage.configureButton).toBeEnabled();
        await modelConfigurationDialogPage.changeEncoderModel('DINOv3 Small');
        expect(modelConfigurationDialogPage.configureButton).toBeDisabled();

        await modelConfigurationDialogPage.changeDecoderModel('SAM2 Small');
        expect(modelConfigurationDialogPage.configureButton).toBeEnabled();
        await modelConfigurationDialogPage.changeDecoderModel('SAM-HQ Tiny');
        expect(modelConfigurationDialogPage.configureButton).toBeDisabled();

        await modelConfigurationDialogPage.changePrecision('FP16');
        expect(modelConfigurationDialogPage.configureButton).toBeEnabled();
        await modelConfigurationDialogPage.changePrecision(model.config.precision.toUpperCase());
        expect(modelConfigurationDialogPage.configureButton).toBeDisabled();

        await modelConfigurationDialogPage.changeUseMaskRefinement();
        expect(modelConfigurationDialogPage.configureButton).toBeEnabled();
        await modelConfigurationDialogPage.changeUseMaskRefinement();
        expect(modelConfigurationDialogPage.configureButton).toBeDisabled();
    });

    it('configures the model', async () => {
        let body: ModelUpdateType;
        server.use(
            http.put('/api/v1/projects/{project_id}/models/{model_id}', async ({ request }) => {
                body = await request.json();

                return HttpResponse.json(request.body);
            })
        );

        const model = getMockedModel();
        const mockOnClose = vi.fn();
        const { modelConfigurationDialogPage } = renderModelConfigurationDialog({ model, onClose: mockOnClose });

        const numberOfForegroundPoints = 20;
        const numberOfBackgroundPoints = 50;
        const confidenceThreshold = 0.8;
        const encoderModel = 'DINOv3 Base';
        const decoderModel = 'SAM2 Small';
        const precision = 'FP16';
        const useMaskRefinement = true;

        await modelConfigurationDialogPage.changeNumberOfForegroundPointes(numberOfForegroundPoints);
        await modelConfigurationDialogPage.changeNumberOfBackgroundPointes(numberOfBackgroundPoints);
        await modelConfigurationDialogPage.changeConfidenceThreshold(confidenceThreshold);
        await modelConfigurationDialogPage.changeEncoderModel(encoderModel);
        await modelConfigurationDialogPage.changeDecoderModel(decoderModel);
        await modelConfigurationDialogPage.changePrecision(precision);
        await modelConfigurationDialogPage.changeUseMaskRefinement();

        await modelConfigurationDialogPage.configureModel();

        await waitFor(() => {
            expect(body).toEqual({
                name: model.name,
                active: model.active,
                config: {
                    model_type: 'matcher',
                    num_foreground_points: numberOfForegroundPoints,
                    num_background_points: numberOfBackgroundPoints,
                    confidence_threshold: confidenceThreshold,
                    sam_model: 'SAM2-small',
                    encoder_model: 'dinov3_base',
                    use_mask_refinement: useMaskRefinement,
                    precision: precision.toLowerCase(),
                },
            });
        });

        expect(mockOnClose).toHaveBeenCalled();
    });

    it('closes the dialog', async () => {
        const onClose = vi.fn();

        const { modelConfigurationDialogPage } = renderModelConfigurationDialog({ onClose });

        await modelConfigurationDialogPage.closeDialog();

        expect(onClose).toHaveBeenCalled();
    });
});
