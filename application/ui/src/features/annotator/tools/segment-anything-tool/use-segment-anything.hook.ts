/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import {
    SegmentAnythingWorkerApi,
    SegmentAnythingWorkerInstance,
} from '@/features/annotator/webworkers/segment-anything.worker.interface';
import { EncodingOutput, SegmentAnythingModel } from '@geti/smart-tools/segment-anything';
import { queryOptions, usePrefetchQuery, useQuery } from '@tanstack/react-query';
import { Remote, wrap } from 'comlink';

import { useAnnotator } from '../../providers/annotator-provider.component';
import { convertToolShapeToGetiShape } from '../utils';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

type SegmentAnythingRemoteInstance = Remote<SegmentAnythingWorkerInstance>;

const segmentAnythingQueryOptions = () =>
    queryOptions<SegmentAnythingRemoteInstance>({
        queryKey: ['workers', 'SEGMENT_ANYTHING'],
        queryFn: async ({ signal }) => {
            const worker = new Worker(new URL('../../webworkers/segment-anything.worker', import.meta.url), {
                type: 'module',
            });

            // addEventListener invokes the listener with `this = signal`, and
            // `Worker.prototype.terminate` requires `this` to be a Worker — it
            // throws "Illegal invocation" silently and the worker is never
            // killed. With React StrictMode double-mounting effects (or any
            // transient cancellation), every cycle leaked a fresh worker, each
            // of which independently re-fetched opencv (~1MB), the ort bundle,
            // ort wasm, and the SAM `.onnx` models. That was the dominant
            // source of the "ort/opencv fetched 6 times" symptom.
            signal.addEventListener('abort', () => worker.terminate(), { once: true });

            try {
                const samWorker = wrap<SegmentAnythingWorkerApi>(worker);
                const instance = await samWorker.build();

                await Promise.all([
                    instance.init('SEGMENT_ANYTHING_ENCODER'),
                    instance.init('SEGMENT_ANYTHING_DECODER'),
                ]);

                if (signal.aborted) {
                    throw signal.reason;
                }

                return instance;
            } catch (error) {
                worker.terminate();

                throw error;
            }
        },
        staleTime: Infinity,
        gcTime: Infinity,
    });

export const usePrefetchSegmentAnythingWorker = () => {
    usePrefetchQuery(segmentAnythingQueryOptions());
};

const useSegmentAnythingWorker = () => {
    const { data } = useQuery(segmentAnythingQueryOptions());

    return data;
};

const useEncodingQuery = (model: Remote<SegmentAnythingModel> | undefined, frameId: string, image: ImageData) => {
    return useQuery({
        queryKey: ['segment-anything-model', 'encoding', frameId],
        queryFn: async () => {
            if (model === undefined) {
                throw new Error('Model not yet initialized');
            }

            if (image === undefined) {
                throw new Error('Image not available');
            }

            return await model.processEncoder(image);
        },
        staleTime: Infinity,
        gcTime: 3600 * 15,
        enabled: model !== undefined,
    });
};

const useDecodingFn = (model: Remote<SegmentAnythingModel> | undefined, encoding: EncodingOutput | undefined) => {
    // TODO: look into returning a new "decoder model" instance that already has the encoding data
    // stored in memory, to reduce  memory usage
    return async (points: InteractiveAnnotationPoint[]) => {
        if (points.length === 0) {
            return [];
        }

        if (model === undefined) {
            return [];
        }

        if (encoding === undefined) {
            return [];
        }

        const { shapes } = await model.processDecoder(encoding, {
            points,
            boxes: [],
            outputConfig: {
                type: 'polygon',
            },
        });

        return shapes.map(convertToolShapeToGetiShape);
    };
};

export const useSegmentAnythingModel = () => {
    const model = useSegmentAnythingWorker();
    const isLoadingWorkers = model === undefined;

    const { frameId, image } = useAnnotator();
    const encodingQuery = useEncodingQuery(model, frameId, image);
    const decodingQueryFn = useDecodingFn(model, encodingQuery.data);

    const isLoading = isLoadingWorkers || encodingQuery.isLoading;
    const isProcessing = encodingQuery.isFetching;

    return {
        isLoading,
        isProcessing,
        encodingQuery,
        decodingQueryFn,
    };
};
