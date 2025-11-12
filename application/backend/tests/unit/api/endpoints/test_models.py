from uuid import uuid4

import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import projects_router
from dependencies import SessionDep, get_model_configuration_service
from domain.errors import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from domain.services.schemas.processor import (
    ProcessorListSchema,
    ProcessorSchema,
)


@pytest.fixture
def app():
    from api.endpoints import models as _  # noqa: F401

    app = FastAPI()
    app.include_router(projects_router, prefix="/api/v1")
    app.dependency_overrides[SessionDep] = lambda: object()

    # Register the global exception handler
    app.add_exception_handler(Exception, custom_exception_handler)
    app.add_exception_handler(RequestValidationError, custom_exception_handler)

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def project_id():
    return uuid4()


@pytest.fixture
def model_id():
    return uuid4()


@pytest.fixture
def sample_processor_schema(model_id):
    return ProcessorSchema(
        id=model_id,
        name="Test Model",
        active=True,
        config={
            "mask_similarity_threshold": 0.38,
            "model_type": "matcher",
            "num_background_points": 2,
            "num_foreground_points": 40,
            "precision": "bf16",
        },
    )


@pytest.fixture
def create_payload():
    return {
        "name": "New Model",
        "id": str(uuid4()),
        "active": True,
        "config": {
            "mask_similarity_threshold": 0.38,
            "model_type": "matcher",
            "num_background_points": 2,
            "num_foreground_points": 40,
            "precision": "bf16",
        },
    }


@pytest.fixture
def update_payload():
    return {
        "name": "Update Model",
        "id": str(uuid4()),
        "active": True,
        "config": {
            "mask_similarity_threshold": 0.38,
            "model_type": "matcher",
            "num_background_points": 2,
            "num_foreground_points": 40,
            "precision": "bf16",
        },
    }


class TestGetModelsConfiguration:
    def test_get_models_configuration_success(self, client, project_id, sample_processor_schema):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def list_model_configurations(self, project_id):
                return ProcessorListSchema(model_configurations=[sample_processor_schema])

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["model_configurations"]) == 1

    def test_get_models_configuration_empty_list(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def list_model_configurations(self, project_id):
                return ProcessorListSchema(model_configurations=[])

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["model_configurations"] == []

    def test_get_models_configuration_project_not_found(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def list_model_configurations(self, project_id):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert str(project_id) in response.json()["detail"]

    def test_get_models_configuration_internal_error(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def list_model_configurations(self, project_id):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetActiveModelConfiguration:
    def test_get_active_model_configuration_success(self, client, project_id, sample_processor_schema):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_active_model_configuration(self, project_id):
                return sample_processor_schema

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/active")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["active"] is True

    def test_get_active_model_configuration_not_found(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_active_model_configuration(self, project_id):
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROCESSOR,
                    message="No active model configuration found for specified project",
                )

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/active")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_active_model_configuration_project_not_found(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_active_model_configuration(self, project_id):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/active")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_active_model_configuration_internal_error(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_active_model_configuration(self, project_id):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/active")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetModelConfiguration:
    def test_get_model_configuration_success(self, client, project_id, model_id, sample_processor_schema):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_model_configuration(self, project_id, model_configuration_id):
                return sample_processor_schema

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == str(model_id)

    def test_get_model_configuration_not_found(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_model_configuration(self, project_id, model_configuration_id):
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROCESSOR, resource_id=str(model_configuration_id)
                )

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_model_configuration_project_not_found(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_model_configuration(self, project_id, model_configuration_id):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_model_configuration_internal_error(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_model_configuration(self, project_id, model_configuration_id):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestCreateModelConfiguration:
    def test_create_model_configuration_success(self, client, project_id, model_id, create_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def create_model_configuration(self, project_id, create_data):
                return ProcessorSchema(
                    id=model_id,
                    name=create_data.name,
                    active=False,
                    config=create_data.config,
                )

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.post(f"/api/v1/projects/{project_id}/models", json=create_payload)

        assert response.status_code == status.HTTP_201_CREATED
        assert "Location" in response.headers
        assert response.json()["name"] == "New Model"

    def test_create_model_configuration_duplicate_name(self, client, project_id, create_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def create_model_configuration(self, project_id, create_data):
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROCESSOR,
                    resource_value=create_data.name,
                    field="name",
                    message="A model configuration with this name already exists in the project.",
                )

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.post(f"/api/v1/projects/{project_id}/models", json=create_payload)

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in response.json()["detail"].lower()

    def test_create_model_configuration_project_not_found(self, client, project_id, create_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def create_model_configuration(self, project_id, create_data):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.post(f"/api/v1/projects/{project_id}/models", json=create_payload)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_model_configuration_internal_error(self, client, project_id, create_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def create_model_configuration(self, project_id, create_data):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.post(f"/api/v1/projects/{project_id}/models", json=create_payload)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestUpdateModelConfiguration:
    def test_update_model_configuration_name_success(
        self, client, project_id, model_id, sample_processor_schema, update_payload
    ):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model_configuration(self, project_id, model_configuration_id, update_data):
                schema = sample_processor_schema.model_copy()
                schema.name = update_data.name
                return schema

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["name"] == "Update Model"

    def test_update_model_configuration_config_success(
        self, client, project_id, model_id, sample_processor_schema, update_payload
    ):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model_configuration(self, project_id, model_configuration_id, update_data):
                schema = sample_processor_schema.model_copy()
                if "config" in update_data:
                    schema.config.update(update_data["config"])
                return schema

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_200_OK

    def test_update_model_configuration_empty_payload(self, client, project_id, model_id, sample_processor_schema):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model_configuration(self, project_id, model_configuration_id, update_data):
                return sample_processor_schema

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        payload = {}
        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=payload)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_update_model_configuration_not_found(self, client, project_id, model_id, update_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model_configuration(self, project_id, model_configuration_id, update_data):
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROCESSOR, resource_id=str(model_configuration_id)
                )

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_model_configuration_project_not_found(self, client, project_id, model_id, update_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model_configuration(self, project_id, model_configuration_id, update_data):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_model_configuration_internal_error(self, client, project_id, model_id, update_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model_configuration(self, project_id, model_configuration_id, update_data):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestDeleteModelConfiguration:
    def test_delete_model_configuration_success(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def delete_model_configuration(self, project_id, model_configuration_id):
                pass

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_model_configuration_not_found(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def delete_model_configuration(self, project_id, model_configuration_id):
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROCESSOR, resource_id=str(model_configuration_id)
                )

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_model_configuration_project_not_found(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def delete_model_configuration(self, project_id, model_configuration_id):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_model_configuration_internal_error(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def delete_model_configuration(self, project_id, model_configuration_id):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_configuration_service] = lambda: FakeProcessorService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
