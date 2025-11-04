/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, minmax, repeat } from '@geti/ui';

import TestImage from '../../../assets/test.webp';
import { ReferencedImage } from './referenced-image/referenced-image.component';

const REFERENCED_IMAGES = [TestImage, TestImage, TestImage, TestImage, TestImage, TestImage];

const useReferencedImages = () => {
    return REFERENCED_IMAGES;
};

export const ReferencedImages = () => {
    const referencedImages = useReferencedImages();

    if (referencedImages === undefined || referencedImages.length === 0) {
        return null;
    }

    return (
        <Grid columns={[repeat('auto-fit', minmax('size-1600', '1fr'))]} gap={'size-100'}>
            {referencedImages.map((image, index) => (
                <ReferencedImage key={index} image={image} />
            ))}
        </Grid>
    );
};
