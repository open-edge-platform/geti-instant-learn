/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

export const loadImage = (link: string): Promise<HTMLImageElement> =>
    new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';

        image.onload = () => resolve(image);
        image.onerror = (error) => reject(error);

        image.fetchPriority = 'high';
        image.src = link;

        if (process.env.NODE_ENV === 'test') {
            // Immediately load the media item's image
            resolve(image);
        }
    });

const drawImageOnCanvas = (img: HTMLImageElement, filter = ''): HTMLCanvasElement => {
    const canvas: HTMLCanvasElement = document.createElement('canvas');

    canvas.width = img.naturalWidth ? img.naturalWidth : img.width;
    canvas.height = img.naturalHeight ? img.naturalHeight : img.height;

    const ctx = canvas.getContext('2d');

    if (ctx) {
        const width = img.naturalWidth ? img.naturalWidth : img.width;
        const height = img.naturalHeight ? img.naturalHeight : img.height;

        ctx.filter = filter;
        ctx.drawImage(img, 0, 0, width, height);
    }

    return canvas;
};

export const getImageData = (img: HTMLImageElement): ImageData => {
    // Always return valid imageData, even if the image isn't loaded yet.
    if (img.width === 0 && img.height === 0) {
        return new ImageData(1, 1);
    }

    const canvas = drawImageOnCanvas(img);
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

    const width = img.naturalWidth ? img.naturalWidth : img.width;
    const height = img.naturalHeight ? img.naturalHeight : img.height;

    return ctx.getImageData(0, 0, width, height);
};
