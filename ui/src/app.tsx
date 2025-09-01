/*
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Button, View } from '@geti/ui';
import { Header } from './components/header.component';

export const App = () => {
  return (
    <View>
      <Header />
      <h1>Rsbuild with React</h1>
      <p>Start building amazing things with Rsbuild.</p>
      <Button>Test</Button>
    </View>
  );
};
