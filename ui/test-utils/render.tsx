import { ReactNode } from 'react';

import { ThemeProvider } from '@geti/ui/theme';
import { RenderOptions, render as rtlRender } from '@testing-library/react';

export const render = (ui: ReactNode, options?: RenderOptions) => {
    return rtlRender(<ThemeProvider>{ui}</ThemeProvider>, options);
};
