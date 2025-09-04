import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';
import tsconfigPaths from 'vite-tsconfig-paths';
import { defineConfig } from 'vitest/config';

export default defineConfig({
    plugins: [
        tsconfigPaths(),
        react(),
        svgr({
            svgrOptions: {
                exportType: 'named',
            },
            include: '**/*.svg',
        }),
    ],
    test: {
        globals: true,
        watch: false,
        environment: 'jsdom',
        include: ['./src/**/*.test.{ts,tsx}'],
        setupFiles: ['./src/setup-test.ts'],
        css: false,
        coverage: {
            provider: 'v8',
            reporter: ['text'],
            reportOnFailure: true,
            include: ['src/**/*.{ts,tsx}'],
        },
        server: {
            deps: {
                inline: [
                    /* CSS files of these packages are not processed by Vite by default, so we get the error that .css file is not
                     * found. To fix it, we need to inline it so Vite can process it. */
                    /@react-spectrum\/.*/,
                    /@spectrum-icons\/.*/,
                    /@adobe\/react-spectrum\/.*/,
                ],
            },
        },
    },
});
