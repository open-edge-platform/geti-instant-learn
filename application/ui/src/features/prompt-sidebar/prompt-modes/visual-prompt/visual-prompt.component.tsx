/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Image } from '@geti-prompt/icons';
import { Button, Flex, View } from '@geti/ui';

import { NoMediaPlaceholder } from '../../../../components/no-media-placeholder/no-media-placeholder.component';
import { CapturedImageLayout } from './captured-image/captured-image-layout.component';
import { ReferencedImages } from './referenced-images.component';
import TestImage from './test.jpg';

const NoImagesPlaceholder = () => {
    return (
        <View minHeight={'size-3000'} height={'50%'} maxHeight={'size-5000'}>
            <NoMediaPlaceholder title={'Capture/Add images for visual prompt'} img={<Image />} />
        </View>
    );
};

const useImage = () => {
    return TestImage;
};

interface VisualPromptContentProps {
    image: string;
}

const VisualPromptContent = ({ image }: VisualPromptContentProps) => {
    return (
        <Flex height={'100%'} direction={'column'} gap={'size-300'}>
            <View flex={1}>
                <CapturedImageLayout image={image} />
            </View>
            <Button alignSelf={'end'} variant={'secondary'}>
                Add to reference images
            </Button>
            <ReferencedImages />
        </Flex>
    );
};

export const VisualPrompt = () => {
    const image = useImage();

    if (image) {
        return <VisualPromptContent image={image} />;
    }

    return <NoImagesPlaceholder />;
};
