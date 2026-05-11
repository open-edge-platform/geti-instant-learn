/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties, PointerEvent, useRef, useState } from 'react';

import { LabelType } from '@/api';
import { clampPointBetweenImage } from '@geti/smart-tools/utils';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useVisualPrompt } from '../../../prompts/visual-prompt/visual-prompt-provider.component';
import { AnnotationShape } from '../../annotations/annotation-shape.component';
import { MaskAnnotations } from '../../annotations/mask-annotations.component';
import { useAnnotationActions } from '../../providers/annotation-actions-provider.component';
import { useAnnotator } from '../../providers/annotator-provider.component';
import { Point, RegionOfInterest, type Annotation as AnnotationType, type Shape } from '../../types';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint, removeOffLimitPoints } from '../utils';
import { CreateLabelPopover } from './create-label.component';
import { SAMLoading } from './sam-loading.component';
import { useSegmentAnythingModel } from './use-segment-anything.hook';
import { useSingleStackFn } from './use-single-stack-fn.hook';

import classes from './segment-anything.module.scss';

interface PreviewAnnotationsProps {
    previewAnnotations: AnnotationType[];
    image: Pick<RegionOfInterest, 'width' | 'height'>;
}

const PreviewAnnotations = ({ previewAnnotations, image }: PreviewAnnotationsProps) => {
    if (previewAnnotations.length === 0) return null;

    return (
        <MaskAnnotations isEnabled annotations={previewAnnotations} width={image.width} height={image.height}>
            {previewAnnotations.map((annotation) => (
                <g
                    key={annotation.id}
                    aria-label='Segment anything preview'
                    style={
                        {
                            '--energy-blue-shade': '#0095ca',
                        } as CSSProperties
                    }
                    stroke={'var(--energy-blue-shade)'}
                    strokeWidth={'calc(3px / var(--zoom-scale))'}
                    fill={'transparent'}
                    className={classes.animateStroke}
                >
                    <AnnotationShape annotation={annotation} />
                </g>
            ))}
        </MaskAnnotations>
    );
};

export const SegmentAnythingTool = () => {
    const [createLabelFormPosition, setCreateLabelFormPosition] = useState<Point | null>(null);
    const [previewShapes, setPreviewShapes] = useState<Shape[]>([]);
    const [acceptedShapes, setAcceptedShapes] = useState<Shape[] | null>(null);
    const ref = useRef<SVGSVGElement>(null);

    const zoom = useZoom();
    const { roi, image } = useAnnotator();
    const { addAnnotations } = useAnnotationActions();
    const { selectedLabel, labels } = useVisualPrompt();
    const { isLoading, isProcessing, isEncodingError, encodingError, retryEncoding, decodingQueryFn } =
        useSegmentAnythingModel();
    const throttledDecodingQueryFn = useSingleStackFn(decodingQueryFn);

    const canvasRef = useRef<SVGRectElement>(null);

    const clampPoint = clampPointBetweenImage(image);

    const handleMouseMove = (event: PointerEvent<SVGSVGElement>) => {
        if (acceptedShapes !== null) {
            return;
        }

        if (!canvasRef.current) {
            return;
        }

        const point = clampPoint(
            getRelativePoint(canvasRef.current, { x: event.clientX, y: event.clientY }, zoom.scale)
        );

        throttledDecodingQueryFn([{ ...point, positive: true }])
            .then((shapes) => {
                setPreviewShapes(shapes.map((shape) => removeOffLimitPoints(shape, roi)));
            })
            .catch((error: unknown) => {
                // Decoding failed – log so it's visible in DevTools
                console.error('[SAM] Decoding error:', error);
                return [];
            });
    };

    const handleAddAnnotations = (shapes: Shape[], label: LabelType) => {
        addAnnotations(shapes, [label]);
        setPreviewShapes([]);
    };

    const handleAddAnnotationsCreateLabel = (label: LabelType) => {
        if (acceptedShapes === null) {
            return;
        }

        handleAddAnnotations(acceptedShapes, label);
        setAcceptedShapes(null);
    };

    const handlePointerDown = (event: PointerEvent<SVGSVGElement>) => {
        if (!ref.current) {
            return;
        }

        if (event.button !== 0 && event.button !== 2) {
            return;
        }

        if (previewShapes.length === 0) {
            return;
        }

        if (selectedLabel == null) {
            const boundingBox = ref.current.getBoundingClientRect();

            const point = {
                x: event.clientX - boundingBox.left,
                y: event.clientY - boundingBox.bottom,
            };

            setCreateLabelFormPosition(point);
            setAcceptedShapes(previewShapes);
            return;
        }

        handleAddAnnotations(previewShapes, selectedLabel);
    };

    const previewAnnotations = (acceptedShapes ?? previewShapes).map((shape, idx): AnnotationType => {
        return {
            shape,
            // During preview mode (while hovering), display the annotation without label color
            // to provide an unobscured view of the underlying image before finalizing placement.
            labels: [],
            id: `${idx}`,
        };
    });

    const handleClose = () => {
        setCreateLabelFormPosition(null);
        setAcceptedShapes(null);
    };

    if (isLoading || isProcessing) {
        return <SAMLoading isLoading={isLoading || isProcessing} />;
    }

    if (isEncodingError) {
        return <SAMLoading isLoading={false} isError errorMessage={String(encodingError)} onRetry={retryEncoding} />;
    }

    return (
        <>
            <SvgToolCanvas
                ref={ref}
                aria-label='SAM tool canvas'
                image={image}
                canvasRef={canvasRef}
                onPointerMove={handleMouseMove}
                onPointerDown={handlePointerDown}
                onPointerLeave={() => {
                    setPreviewShapes([]);
                }}
                style={{
                    cursor: `url("/icons/selection.svg") 8 8, auto`,
                }}
            >
                <PreviewAnnotations previewAnnotations={previewAnnotations} image={image} />
            </SvgToolCanvas>
            <CreateLabelPopover
                ref={ref}
                onSuccess={handleAddAnnotationsCreateLabel}
                existingLabels={labels}
                mousePosition={createLabelFormPosition}
                onClose={handleClose}
            />
        </>
    );
};
