/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@/test-utils';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import * as filePicker from '../../../../shared/tauri/file-picker';
import { VideoFileFields } from './video-file-fields.component';

// Mock the file picker
vi.mock('../../../../shared/tauri/file-picker', () => ({
    pickFilePath: vi.fn(),
    pickFolderPath: vi.fn(),
    isTauriRuntime: vi.fn(),
}));

describe('VideoFileFields', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('Browse button visibility and behavior', () => {
        it('renders file path text field', () => {
            const onChange = vi.fn();
            render(<VideoFileFields filePath='' onFilePathChange={onChange} />);

            expect(screen.getByRole('textbox', { name: /File path/ })).toBeInTheDocument();
        });

        it('renders browse button', () => {
            const onChange = vi.fn();
            render(<VideoFileFields filePath='' onFilePathChange={onChange} />);

            expect(screen.getByRole('button', { name: /Browse/ })).toBeInTheDocument();
        });

        it('disables browse button when not in Tauri runtime', () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(false);
            const onChange = vi.fn();
            render(<VideoFileFields filePath='' onFilePathChange={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            expect(browseButton).toBeDisabled();
        });

        it('enables browse button when in Tauri runtime', () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(true);
            const onChange = vi.fn();
            render(<VideoFileFields filePath='' onFilePathChange={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            expect(browseButton).not.toBeDisabled();
        });
    });

    describe('File picker integration', () => {
        it('opens file picker when browse button is clicked in Tauri', async () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(true);
            vi.mocked(filePicker.pickFilePath).mockResolvedValue(null);

            const onChange = vi.fn();
            render(<VideoFileFields filePath='' onFilePathChange={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            await userEvent.click(browseButton);

            expect(vi.mocked(filePicker.pickFilePath)).toHaveBeenCalled();
        });

        it('updates file path when file is selected', async () => {
            const testFilePath = '/home/user/video.mp4';
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(true);
            vi.mocked(filePicker.pickFilePath).mockResolvedValue(testFilePath);

            const onChange = vi.fn();
            render(<VideoFileFields filePath='' onFilePathChange={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            await userEvent.click(browseButton);

            expect(onChange).toHaveBeenCalledWith(testFilePath);
        });

        it('does not update file path when picker returns null', async () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(true);
            vi.mocked(filePicker.pickFilePath).mockResolvedValue(null);

            const onChange = vi.fn();
            render(<VideoFileFields filePath='/existing/path.mp4' onFilePathChange={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            await userEvent.click(browseButton);

            expect(onChange).not.toHaveBeenCalled();
        });

        it('does not call file picker when browse button is disabled (web runtime)', async () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(false);
            vi.mocked(filePicker.pickFilePath).mockResolvedValue(null);

            const onChange = vi.fn();
            render(<VideoFileFields filePath='' onFilePathChange={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });

            // Try to click disabled button - it should not be clickable
            expect(browseButton).toBeDisabled();
            await userEvent.click(browseButton);

            expect(vi.mocked(filePicker.pickFilePath)).not.toHaveBeenCalled();
        });
    });

    describe('File path input', () => {
        it('allows manual entry of file path', async () => {
            const onChange = vi.fn();
            render(<VideoFileFields filePath='' onFilePathChange={onChange} />);

            const input = screen.getByRole('textbox', { name: /File path/ });
            await userEvent.type(input, '/path/to/file.mp4');

            expect(onChange).toHaveBeenCalled();
        });

        it('displays provided file path', () => {
            const testPath = '/home/user/video.mp4';
            const onChange = vi.fn();
            render(<VideoFileFields filePath={testPath} onFilePathChange={onChange} />);

            expect(screen.getByRole('textbox', { name: /File path/ })).toHaveValue(testPath);
        });
    });
});
