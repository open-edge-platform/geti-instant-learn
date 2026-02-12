import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import HomepageFeatures from '@site/src/components/home-page-features';
import { ModelLifecycle } from '@site/src/components/model-lifecycle/model-lifecycle';
import { ModelOptimizationDeployment } from '@site/src/components/model-optimization-deployment/model-optimization-deployment';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import ThemedImage from '@theme/ThemedImage';
import clsx from 'clsx';

import styles from './index.module.css';

function HomepageHeader() {
    const { siteConfig } = useDocusaurusContext();

    return (
        <header className={clsx('hero', styles.heroBanner)}>
            <div className={styles.bannerContent}>
                <div className={styles.bannerLogo}>
                    <ThemedImage
                        sources={{
                            light: '/img/light/geti-logo-big-light.svg',
                            dark: '/img/dark/geti-logo-big-dark.svg',
                        }}
                    />
                </div>
                <div className={styles.bannerTextContent}>
                    <Heading as='h1' className={styles.title}>
                        {siteConfig.tagline}
                    </Heading>
                    <p>
                        As part of Intel's Open Edge Platform, Geti™ enables anyone from domain experts to data
                        scientists to rapidly develop production-ready AI models.
                    </p>
                </div>
                <div className={styles.oep}>
                    <Heading as='h2'>
                        <Link to='https://www.intel.com/content/www/us/en/developer/tools/tiber/edge-platform/overview.html'>
                            Open Edge Platform
                        </Link>
                    </Heading>
                </div>
            </div>
        </header>
    );
}

export default function Home(): JSX.Element {
    const { siteConfig } = useDocusaurusContext();

    return (
        <Layout title={siteConfig.tagline} description={siteConfig.tagline}>
            <main className={styles.main}>
                <HomepageHeader />
                <HomepageFeatures />
                <ModelLifecycle />
                <ModelOptimizationDeployment />
            </main>
        </Layout>
    );
}
