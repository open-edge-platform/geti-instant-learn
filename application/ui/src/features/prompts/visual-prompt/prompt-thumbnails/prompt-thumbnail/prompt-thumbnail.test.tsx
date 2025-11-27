/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { VisualPromptItemType, VisualPromptType } from '@geti-prompt/api';
import { render } from '@geti-prompt/test-utils';
import { fireEvent, screen, waitFor, waitForElementToBeRemoved } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http, server } from '../../../../../setup-test';
import { SelectedFrameProvider, useSelectedFrame } from '../../../../../shared/selected-frame-provider.component';
import { useVisualPrompt, VisualPromptProvider } from '../../visual-prompt-provider.component';
import { PromptThumbnail } from './prompt-thumbnail.component';

const getMockedPrompt = (prompt: Partial<VisualPromptType> = {}): VisualPromptType => {
    return {
        id: '123',
        frame_id: '321',
        type: 'VISUAL',
        annotations: [
            {
                label_id: '123',
                config: {
                    type: 'polygon',
                    points: [],
                },
            },
        ],
        ...prompt,
    };
};

const getMockedPromptItem = (prompt: Partial<VisualPromptItemType> = {}): VisualPromptItemType => {
    return {
        ...getMockedPrompt(prompt),
        thumbnail: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA',
        ...prompt,
    };
};

class PromptThumbnailPage {
    openMenuActions(id: string) {
        fireEvent.click(screen.getByRole('button', { name: `Prompt actions ${id}` }));
    }

    edit(id: string) {
        this.openMenuActions(id);
        fireEvent.click(screen.getByRole('menuitem', { name: /Edit/i }));
    }

    delete(id: string) {
        this.openMenuActions(id);
        fireEvent.click(screen.getByRole('menuitem', { name: /Delete/i }));
    }

    getThumbnail(id: string) {
        return screen.getByLabelText(`prompt thumbnail ${id}`);
    }
}

const App = ({ promptItem }: { promptItem: VisualPromptItemType }) => {
    const { selectedFrameId } = useSelectedFrame();
    const { promptId, prompt: fetchedPrompt } = useVisualPrompt();

    return (
        <>
            <span aria-label={'Selected frame id'}>{selectedFrameId ?? 'Empty'}</span>
            <span aria-label={'Selected prompt id'}>{promptId ?? 'Empty'}</span>
            <span aria-label={'Prompt'}>{fetchedPrompt ? 'loaded' : 'loading'}</span>
            <PromptThumbnail prompt={promptItem} />
        </>
    );
};

const renderPromptThumbnail = async ({
    frameId,
    promptItem,
}: {
    frameId: string;
    promptItem: VisualPromptItemType;
}) => {
    const result = render(
        <SelectedFrameProvider frameId={frameId}>
            <VisualPromptProvider>
                <App promptItem={promptItem} />
            </VisualPromptProvider>
        </SelectedFrameProvider>
    );

    await waitForElementToBeRemoved(screen.getByRole('progressbar'));

    return {
        result,
        promptThumbnailPage: new PromptThumbnailPage(),
    };
};

describe('PromptThumbnail', () => {
    it('deletes the prompt', async () => {
        let promptIdToBeDeleted: string | null = null;
        const prompt = getMockedPrompt();
        const promptThumbnail = getMockedPromptItem({ ...prompt });

        server.use(
            http.get('/api/v1/projects/{project_id}/prompts/{prompt_id}', () => {
                return HttpResponse.json({
                    ...prompt,
                });
            }),

            http.get('/api/v1/projects/{project_id}/prompts', () => {
                return HttpResponse.json({
                    prompts: [promptThumbnail],
                    pagination: {
                        count: 1,
                        total: 1,
                        offset: 0,
                        limit: 10,
                    },
                });
            }),

            http.delete('/api/v1/projects/{project_id}/prompts/{prompt_id}', ({ params }) => {
                promptIdToBeDeleted = params.prompt_id;

                return HttpResponse.json({}, { status: 204 });
            })
        );

        const frameId = 'frame-123';

        const { promptThumbnailPage } = await renderPromptThumbnail({
            frameId,
            promptItem: getMockedPromptItem({ id: prompt.id }),
        });

        expect(promptThumbnailPage.getThumbnail(prompt.id)).toBeInTheDocument();

        promptThumbnailPage.delete(prompt.id);

        await waitFor(() => {
            expect(promptIdToBeDeleted).toBe(prompt.id);
        });

        expect(screen.getByLabelText('Selected frame id')).toHaveTextContent(frameId);
        expect(screen.getByLabelText('Selected prompt id')).toHaveTextContent('Empty');
    });

    it('deletes the prompt and resets the selected frame id when that frame is in edition', async () => {
        let promptIdToBeDeleted: string | null = null;
        const prompt = getMockedPrompt();
        const promptThumbnail = getMockedPromptItem({ ...prompt });

        server.use(
            http.get('/api/v1/projects/{project_id}/prompts/{prompt_id}', () => {
                return HttpResponse.json({
                    ...prompt,
                });
            }),

            http.get('/api/v1/projects/{project_id}/prompts', () => {
                return HttpResponse.json({
                    prompts: [promptThumbnail],
                    pagination: {
                        count: 1,
                        total: 1,
                        offset: 0,
                        limit: 10,
                    },
                });
            }),

            http.delete('/api/v1/projects/{project_id}/prompts/{prompt_id}', ({ params }) => {
                promptIdToBeDeleted = params.prompt_id;

                return HttpResponse.json({}, { status: 204 });
            })
        );

        const frameId = 'frame-123';

        const { promptThumbnailPage } = await renderPromptThumbnail({
            frameId,
            promptItem: getMockedPromptItem({ id: prompt.id }),
        });

        expect(promptThumbnailPage.getThumbnail(prompt.id)).toBeInTheDocument();

        promptThumbnailPage.edit(prompt.id);

        await waitFor(() => {
            expect(screen.getByLabelText('Prompt')).toHaveTextContent('loaded');
            expect(screen.getByLabelText('Selected frame id')).toHaveTextContent(prompt.frame_id);
        });

        promptThumbnailPage.delete(prompt.id);

        await waitFor(() => {
            expect(promptIdToBeDeleted).toBe(prompt.id);
        });

        expect(screen.getByLabelText('Selected frame id')).toHaveTextContent('Empty');
        expect(screen.getByLabelText('Selected prompt id')).toHaveTextContent('Empty');
    });

    it('sets correct id when editing', async () => {
        const prompt = getMockedPrompt();
        const promptThumbnail = getMockedPromptItem({ ...prompt });

        server.use(
            http.get('/api/v1/projects/{project_id}/prompts/{prompt_id}', () => {
                return HttpResponse.json({
                    ...prompt,
                });
            }),

            http.get('/api/v1/projects/{project_id}/prompts', () => {
                return HttpResponse.json({
                    prompts: [promptThumbnail],
                    pagination: {
                        count: 1,
                        total: 1,
                        offset: 0,
                        limit: 10,
                    },
                });
            })
        );

        const { promptThumbnailPage } = await renderPromptThumbnail({
            frameId: 'frame-123',
            promptItem: getMockedPromptItem({ id: prompt.id }),
        });

        expect(promptThumbnailPage.getThumbnail(prompt.id)).toBeInTheDocument();

        promptThumbnailPage.edit(prompt.id);

        await waitFor(() => {
            expect(screen.getByLabelText('Prompt')).toHaveTextContent('loaded');
            expect(screen.getByLabelText('Selected frame id')).toHaveTextContent(prompt.frame_id);
            expect(screen.getByLabelText('Selected prompt id')).toHaveTextContent(prompt.id);
        });
    });
});
