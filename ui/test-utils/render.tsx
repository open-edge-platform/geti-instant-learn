import { render as rtlRender, RenderOptions } from '@testing-library/react';
import { ReactNode } from 'react';
import { ThemeProvider } from '@geti/ui/theme';

export const render = (ui: ReactNode, options?: RenderOptions) => {
  return rtlRender(<ThemeProvider>{ui}</ThemeProvider>, options);
};
