# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ResourceType(str, Enum):
    """Enumeration for resource types."""

    ANNOTATION = "Annotation"
    LABEL = "Label"
    PROMPT = "Prompt"
    SOURCE = "Source"
    PROCESSOR = "Processor"
    SINK = "Sink"
    PROJECT = "Project"


class ResourceError(Exception):
    """Base exception for resource-related errors."""

    def __init__(self, resource_type: ResourceType, resource_id: str | None, message: str):
        super().__init__(message)
        self.resource_type: ResourceType = resource_type
        self.resource_id: str | None = resource_id


class ResourceNotFoundError(ResourceError):
    """Exception raised when a resource is not found."""

    def __init__(self, resource_type: ResourceType, resource_id: str | None = None, message: str | None = None):
        msg = message or f"{resource_type.value} with ID {resource_id} not found."
        super().__init__(resource_type, resource_id, msg)


class ResourceInUseError(ResourceError):
    """Exception raised when trying to delete a resource that is currently in use."""

    def __init__(self, resource_type: ResourceType, resource_id: str, message: str | None = None):
        msg = message or f"{resource_type.value} with ID {resource_id} cannot be deleted because it is in use."
        super().__init__(resource_type, resource_id, msg)


class ResourceAlreadyExistsError(ResourceError):
    """Exception raised when a resource with the same name or id already exists."""

    def __init__(
        self, resource_type: ResourceType, resource_value: str, raised_by: str = "name", message: str | None = None
    ):
        if not message:
            if raised_by == "id":
                msg = f"{resource_type.value} with id '{resource_value}' already exists."
            else:
                msg = f"{resource_type.value} with name '{resource_value}' already exists."
        else:
            msg = message
        super().__init__(resource_type, resource_value, msg)


class ResourceUpdateConflictError(ResourceError):
    """Exception raised when attempting to modify an immutable attribute of a resource."""

    def __init__(self, resource_type: ResourceType, resource_id: str, field: str, message: str | None = None):
        msg = message or f"{resource_type.value} with ID {resource_id} cannot change immutable field '{field}'."
        super().__init__(resource_type, resource_id, msg)
        self.field = field
