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
    FRAME = "Frame"


class ServiceError(Exception):
    """Base exception for service-related errors."""


class ResourceError(ServiceError):
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
        self,
        resource_type: ResourceType,
        resource_value: str | None = None,
        field: str = "name",
        message: str | None = None,
    ):
        """
        Initialize ResourceAlreadyExistsError.

        Args:
            resource_type: Type of resource (e.g., PROJECT, SOURCE)
            resource_value: The actual value that caused the conflict (e.g., "My Project")
            field: The field that caused the conflict (e.g., "name", "active")
            message: Custom error message. If not provided, generates a default message.
        """
        if not message:
            if field == "id":
                msg = f"{resource_type.value} with ID '{resource_value}' already exists."
            elif resource_value:
                msg = f"{resource_type.value} with {field} '{resource_value}' already exists in this context."
            else:
                msg = f"{resource_type.value} constraint violation: {field} must be unique."
        else:
            msg = message
        super().__init__(resource_type, resource_value, msg)
        self.field = field


class ResourceUpdateConflictError(ResourceError):
    """Exception raised when attempting to modify an immutable attribute of a resource."""

    def __init__(self, resource_type: ResourceType, resource_id: str, field: str, message: str | None = None):
        msg = message or f"{resource_type.value} with ID {resource_id} cannot change immutable field '{field}'."
        super().__init__(resource_type, resource_id, msg)
        self.field = field
