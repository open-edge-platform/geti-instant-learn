import { render } from '@geti-prompt/test-utils';
import { Header } from './header.component';
import { screen } from '@testing-library/react';

describe('Header', () => {
  it('renders header properly', () => {
    render(<Header />);

    expect(screen.getByText('Geti Prompt')).toBeInTheDocument();
  });
});
