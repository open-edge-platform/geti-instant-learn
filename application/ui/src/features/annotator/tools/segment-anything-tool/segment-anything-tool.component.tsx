/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { CSSProperties, PointerEvent, useEffect, useRef, useState } from 'react';

import { LabelType } from '@geti-prompt/api';
import { clampPointBetweenImage } from '@geti/smart-tools/utils';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useVisualPrompt } from '../../../prompts/visual-prompt/visual-prompt-provider.component';
import { AnnotationShape } from '../../annotations/annotation-shape.component';
import { MaskAnnotations } from '../../annotations/mask-annotations.component';
import { useAnnotationActions } from '../../providers/annotation-actions-provider.component';
import { useAnnotator } from '../../providers/annotator-provider.component';
import { RegionOfInterest, type Annotation as AnnotationType, type Shape } from '../../types';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint, removeOffLimitPoints } from '../utils';
import { CreateLabel, useCreateLabelFormPosition } from './create-label.component';
import { SAMLoading } from './sam-loading.component';
import { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useSegmentAnythingModel } from './use-segment-anything.hook';
import { useSingleStackFn } from './use-single-stack-fn.hook';
import { useThrottledCallback } from './use-throttle-callback.hook';

import classes from './segment-anything.module.scss';

// Whenever the user moves their mouse over the canvas, we compute a preview of
// SAM being applied to the user's mouse position.
// The decoding step of SAM takes on average 100ms with 150-250ms being a high
// exception. We throttle the mouse update based on this so that we don't overload
// the user's cpu with too many decoding requests
const THROTTLE_TIME = 150;

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
    const [mousePosition, setMousePosition] = useState<InteractiveAnnotationPoint>();
    const [createLabelFormPosition, setCreateLabelFormPosition] = useCreateLabelFormPosition();
    const [previewShapes, setPreviewShapes] = useState<Shape[]>([]);
    const [acceptedShapes, setAcceptedShapes] = useState<Shape[] | null>(null);

    const zoom = useZoom();
    const { roi, image } = useAnnotator();
    const { addAnnotations } = useAnnotationActions();
    const { selectedLabel, labels } = useVisualPrompt();
    const { isLoading, decodingQueryFn } = useSegmentAnythingModel();
    const throttledDecodingQueryFn = useSingleStackFn(decodingQueryFn);

    const canvasRef = useRef<SVGRectElement>(null);

    const clampPoint = clampPointBetweenImage(image);

    const throttleSetMousePosition = useThrottledCallback((point: InteractiveAnnotationPoint) => {
        setMousePosition(point);
    }, THROTTLE_TIME);

    useEffect(() => {
        if (mousePosition === undefined) {
            return;
        }

        throttledDecodingQueryFn([mousePosition])
            .then((shapes) => {
                setPreviewShapes(shapes.map((shape) => removeOffLimitPoints(shape, roi)));

                throttleSetMousePosition.flush();
            })
            .catch(() => {
                // If getting decoding went wrong we set an empty preview and
                // start to compute the next decoding
                return [];
            });
    }, [mousePosition, throttledDecodingQueryFn, throttleSetMousePosition, roi]);

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

        throttleSetMousePosition({ ...point, positive: true });
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
        if (!canvasRef.current) {
            return;
        }

        if (event.button !== 0 && event.button !== 2) {
            return;
        }

        if (previewShapes.length === 0) {
            return;
        }

        if (selectedLabel == null) {
            setCreateLabelFormPosition({ x: event.clientX, y: event.clientY });
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
        setCreateLabelFormPosition(undefined);
        setAcceptedShapes(null);
    };

    if (isLoading) {
        return <SAMLoading isLoading={isLoading} />;
    }

    return (
        <>
            <SvgToolCanvas
                aria-label='SAM tool canvas'
                image={image}
                canvasRef={canvasRef}
                onPointerMove={handleMouseMove}
                onPointerDown={handlePointerDown}
                onPointerLeave={() => {
                    throttleSetMousePosition.cancel();
                    setMousePosition(undefined);
                    setPreviewShapes([]);
                }}
                style={{
                    cursor: `url("/icons/selection.svg") 8 8, auto`,
                }}
            >
                <PreviewAnnotations previewAnnotations={previewAnnotations} image={image} />
            </SvgToolCanvas>
            <CreateLabel
                onSuccess={handleAddAnnotationsCreateLabel}
                existingLabels={labels}
                mousePosition={createLabelFormPosition}
                onClose={handleClose}
            />
        </>
    );
};
