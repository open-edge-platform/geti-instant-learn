export const GETI_INSTALLER_LOCATION =
    'https://storage.geti.intel.com/$(curl -L -s https://storage.geti.intel.com/latest.txt)/platform_installer.tar.gz';

export const DEEP_LEARNING_COMPONENTS = [
    {
        title: 'Anomalib',
        description:
            'Deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation of custom models.',
        docHref: 'https://anomalib.readthedocs.io/en/v1.1.3/',
        githubHref: 'https://github.com/open-edge-platform/anomalib',
    },
    {
        title: 'Geti™ SDK',
        description:
            'Software Development Kit (SDK) for the Geti™ platform for Computer Vision AI model training.',
        docHref: 'https://open-edge-platform.github.io/geti-sdk/',
        githubHref: 'https://github.com/open-edge-platform/geti-sdk',
    },
];
