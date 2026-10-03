import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';

export default function Home(): JSX.Element {
  return (
    <Layout
      title="Geti Instant Learn"
      description="Documentation hub for the Geti Instant Learn library and application"
    >
      <main>
        <section className="hero hero--primary">
          <div className="container">
            <h1 className="hero__title">Geti Instant Learn</h1>
            <p className="hero__subtitle">
              Explore the library, application, and concepts behind visual prompting.
            </p>
            <div className="margin-top--md">
              <Link className="button button--secondary button--lg margin-right--md" to="/docs/library/introduction">
                Library Docs
              </Link>
              <Link className="button button--outline button--lg" to="/docs/application/introduction">
                Application Docs
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}