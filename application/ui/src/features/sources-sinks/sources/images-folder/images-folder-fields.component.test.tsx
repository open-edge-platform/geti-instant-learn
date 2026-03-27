/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from '@/test-utils';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import * as filePicker from '../../../../shared/tauri/file-picker';
import { ImagesFolderFields } from './images-folder-fields.component';

// Mock the file picker
vi.mock('../../../../shared/tauri/file-picker', () => ({
    pickVideoFilePath: vi.fn(),
    pickFolderPath: vi.fn(),
    isTauriRuntime: vi.fn(),
}));

describe('ImagesFolderFields', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('Browse button visibility and behavior', () => {
        it('renders folder path text field', () => {
            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='' onSetFolderPath={onChange} />);

            expect(screen.getByRole('textbox', { name: /Folder path/ })).toBeInTheDocument();
        });

        it('renders browse button', () => {
            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='' onSetFolderPath={onChange} />);

            expect(screen.getByRole('button', { name: /Browse/ })).toBeInTheDocument();
        });

        it('disables browse button when not in Tauri runtime', () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(false);
            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='' onSetFolderPath={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            expect(browseButton).toBeDisabled();
        });

        it('enables browse button when in Tauri runtime', () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(true);
            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='' onSetFolderPath={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            expect(browseButton).not.toBeDisabled();
        });
    });

    describe('Folder picker integration', () => {
        it('opens folder picker when browse button is clicked in Tauri', async () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(true);
            vi.mocked(filePicker.pickFolderPath).mockResolvedValue(null);

            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='' onSetFolderPath={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            await userEvent.click(browseButton);

            expect(vi.mocked(filePicker.pickFolderPath)).toHaveBeenCalled();
        });

        it('updates folder path when folder is selected', async () => {
            const testFolderPath = '/home/user/images';
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(true);
            vi.mocked(filePicker.pickFolderPath).mockResolvedValue(testFolderPath);

            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='' onSetFolderPath={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            await userEvent.click(browseButton);

            expect(onChange).toHaveBeenCalledWith(testFolderPath);
        });

        it('does not update folder path when picker returns null', async () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(true);
            vi.mocked(filePicker.pickFolderPath).mockResolvedValue(null);

            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='/existing/path' onSetFolderPath={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });
            await userEvent.click(browseButton);

            expect(onChange).not.toHaveBeenCalled();
        });

        it('does not call folder picker when browse button is disabled (web runtime)', async () => {
            vi.mocked(filePicker.isTauriRuntime).mockReturnValue(false);
            vi.mocked(filePicker.pickFolderPath).mockResolvedValue(null);

            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='' onSetFolderPath={onChange} />);

            const browseButton = screen.getByRole('button', { name: /Browse/ });

            // Try to click disabled button - it should not be clickable
            expect(browseButton).toBeDisabled();
            await userEvent.click(browseButton);

            expect(vi.mocked(filePicker.pickFolderPath)).not.toHaveBeenCalled();
        });
    });

    describe('Folder path input', () => {
        it('allows manual entry of folder path', async () => {
            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='' onSetFolderPath={onChange} />);

            const input = screen.getByRole('textbox', { name: /Folder path/ });
            await userEvent.type(input, '/path/to/folder');

            expect(onChange).toHaveBeenCalled();
        });

        it('displays provided folder path', () => {
            const testPath = '/home/user/images';
            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath={testPath} onSetFolderPath={onChange} />);

            expect(screen.getByRole('textbox', { name: /Folder path/ })).toHaveValue(testPath);
        });
    });

    describe('Contextual help', () => {
        it('displays contextual help for folder path', () => {
            const onChange = vi.fn();
            render(<ImagesFolderFields folderPath='' onSetFolderPath={onChange} />);

            // The contextual help should be rendered (it's an icon that opens a popover)
            const contextualHelpButton = screen.getByRole('button', { name: /Information/ });
            expect(contextualHelpButton).toBeInTheDocument();
        });
    });
});
