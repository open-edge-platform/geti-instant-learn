# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session


class BaseRepository:
    def __init__(self, session: Session):
        """Initialize the repository"""
        self.session = session
