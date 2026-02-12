import { LearnMore } from '@site/src/components/go-to-link/learn-more';
import { Section } from '@site/src/components/section/section';
import TabItem from '@theme/TabItem';
import Tabs from '@theme/Tabs';
import ThemedImage from '@theme/ThemedImage';

import { AutoPlayVideo } from '../auto-play-video/auto-play-video';

const TABS = [
    {
        label: 'Code deployment',
        videoURL:
            'https://s3-figma-videos-production-sig.figma.com/video/954166258663478463/ORG/3e39/5510/-1b86-4bba-a322-56f2037c6c58?Expires=1734912000&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=g5swCsVykhUAz-fzXLTy1G5wqte8LHltAMziHQBgOn0MJwi8EjYEpfFNZGMkd4H0bkyMn71aVDPiWKvqcXzDYJvya5xHFwtD6K050MSw5tIQdvbXEm~-3QZ7Uv~5~boEZbIaje7ymJX9UyJlDvgm0PYURVIfqMgdVZcdUlAzAws2fA7wWiKxeaJfEEDOc0DpeARhtkcaFMiMcIDntImfL5Kg65toYpP6OCZMzGJ719qIgKeziSdL2AF76lVkWmGGeK837DUvzgSJrZ1gxu3vvrrsflkvB7mQ~AO9hyJklviy3wd1H7aZz-2xUWWRPzEdNR34aCte1muZrVQytDzTsw__',
        description:
            'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
        urlLink: '/',
    },
    {
        label: 'Test Drive',
        videoURL:
            'https://s3-figma-videos-production-sig.figma.com/video/954166258663478463/ORG/3e39/5510/-1b86-4bba-a322-56f2037c6c58?Expires=1734912000&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=g5swCsVykhUAz-fzXLTy1G5wqte8LHltAMziHQBgOn0MJwi8EjYEpfFNZGMkd4H0bkyMn71aVDPiWKvqcXzDYJvya5xHFwtD6K050MSw5tIQdvbXEm~-3QZ7Uv~5~boEZbIaje7ymJX9UyJlDvgm0PYURVIfqMgdVZcdUlAzAws2fA7wWiKxeaJfEEDOc0DpeARhtkcaFMiMcIDntImfL5Kg65toYpP6OCZMzGJ719qIgKeziSdL2AF76lVkWmGGeK837DUvzgSJrZ1gxu3vvrrsflkvB7mQ~AO9hyJklviy3wd1H7aZz-2xUWWRPzEdNR34aCte1muZrVQytDzTsw__',
        description:
            'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
        urlLink: '/',
    },
    {
        label: 'Inference Server Deployment',
        videoURL:
            'https://s3-figma-videos-production-sig.figma.com/video/954166258663478463/ORG/3e39/5510/-1b86-4bba-a322-56f2037c6c58?Expires=1734912000&Key-Pair-Id=APKAQ4GOSFWCVNEHN3O4&Signature=g5swCsVykhUAz-fzXLTy1G5wqte8LHltAMziHQBgOn0MJwi8EjYEpfFNZGMkd4H0bkyMn71aVDPiWKvqcXzDYJvya5xHFwtD6K050MSw5tIQdvbXEm~-3QZ7Uv~5~boEZbIaje7ymJX9UyJlDvgm0PYURVIfqMgdVZcdUlAzAws2fA7wWiKxeaJfEEDOc0DpeARhtkcaFMiMcIDntImfL5Kg65toYpP6OCZMzGJ719qIgKeziSdL2AF76lVkWmGGeK837DUvzgSJrZ1gxu3vvrrsflkvB7mQ~AO9hyJklviy3wd1H7aZz-2xUWWRPzEdNR34aCte1muZrVQytDzTsw__',
        description:
            'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
        urlLink: '/',
    },
];

const ModelDeploymentTabs = () => {
    return (
        <Tabs>
            {TABS.map((tab) => (
                <TabItem value={tab.label} key={tab.label}>
                    <Section.Row>
                        <Section.Column flex={2}>
                            <AutoPlayVideo videoUrl={tab.videoURL} />
                        </Section.Column>
                        <Section.Column>
                            <Section.Description>{tab.description}</Section.Description>
                            <Section.Divider />
                            <LearnMore link={tab.urlLink} />
                        </Section.Column>
                    </Section.Row>
                </TabItem>
            ))}
        </Tabs>
    );
};

export const ModelOptimizationDeployment = () => {
    return (
        <Section withBackground withPadding>
            <Section.Title>Model Optimization & Deployment</Section.Title>
            <Section.Description marginBottom={'1rem'}>
                Under the hood, Geti™ utilizes the OpenVINO™ toolkit to optimize the model performance for deployment
                across the whole Intel hardware portfolio in different quantizations so that you can choose the
                precision that fits your use case: INT8, FP16, and FP32.
            </Section.Description>
            <Section.Row justifyContent={'center'} alignItems={'center'}>
                <ThemedImage
                    alt={'Geti™ Optimization'}
                    sources={{
                        light: '/img/light/geti-optimization-light.svg',
                        dark: '/img/dark/geti-optimization-dark.svg',
                    }}
                    style={{ marginTop: '1rem' }}
                />
            </Section.Row>
            <Section.Description marginTop={'1.5rem'}>
                Once your model is ready for integration, Geti™ offers a range of deployment options to put your model
                to work in real-time. Whether you want to easily run inference in our dedicated{' '}
                <a
                    href='https://github.com/openvinotoolkit/openvino_testdrive'
                    target='_blank'
                    rel='noopener noreferrer'
                >
                    Test Drive GUI
                </a>
                , run your application as an inference server, or utilize the{' '}
                <a href='https://github.com/open-edge-platform/geti-sdk' target='_blank' rel='noopener noreferrer'>
                    Geti™ SDK
                </a>
                : with Geti™, you can easily put your model to action with the deployment option that meets your needs.
            </Section.Description>

            {/* TODO: Hide this section until we gather all the videos */}
            {/* <ModelDeploymentTabs /> */}
        </Section>
    );
};
