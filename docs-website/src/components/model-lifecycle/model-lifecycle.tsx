import { Section } from '@site/src/components/section/section';
import ThemedImage from '@theme/ThemedImage';

export const ModelLifecycle = () => {
    return (
        <Section>
            <Section.Title as={'h1'} center>
                Geti™ Model Lifecycle
            </Section.Title>
            <Section.Description marginBottom={'1.5rem'} center>
                Geti™ enables anyone to build computer vision AI models in a fraction of the time and with
                minimal data.
                <br />
                The software provides you with a seamless, end-to-end workflow to prepare state-of-the-art
                <br /> computer vision models in minutes.
            </Section.Description>
            <Section.Row justifyContent={'center'}>
                <ThemedImage
                    alt={'Model lifecycle'}
                    sources={{
                        light: '/img/light/model-lifecycle-infinity-light.png',
                        dark: '/img/dark/model-lifecycle-infinity-dark.png',
                    }}
                />
            </Section.Row>
        </Section>
    );
};
