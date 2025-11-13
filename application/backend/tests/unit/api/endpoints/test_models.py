from uuid import uuid4

import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import projects_router
from dependencies import SessionDep, get_model_service
from domain.errors import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from domain.services.schemas.base import Pagination
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


class TestGetModels:
    def test_get_models_success(self, client, project_id, sample_processor_schema):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def list_models(self, project_id, offset=0, limit=100):
                items = [sample_processor_schema]
                pagination = Pagination(
                    count=len(items),
                    total=1,
                    offset=offset,
                    limit=limit,
                )
                return ProcessorListSchema(models=[sample_processor_schema], pagination=pagination)

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["models"]) == 1

    def test_get_models_empty_list(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def list_models(self, project_id, offset=0, limit=100):
                pagination = Pagination(count=0, total=0, offset=offset, limit=limit)
                return ProcessorListSchema(models=[], pagination=pagination)

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["models"] == []

    def test_get_models_project_not_found(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def list_models(self, project_id, offset=0, limit=100):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert str(project_id) in response.json()["detail"]

    def test_get_models_internal_error(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def list_models(self, project_id, offset=0, limit=100):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetActiveModel:
    def test_get_active_model_success(self, client, project_id, sample_processor_schema):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_active_model(self, project_id):
                return sample_processor_schema

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/active")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["active"] is True

    def test_get_active_model_not_found(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_active_model(self, project_id):
                raise ResourceNotFoundError(
                    resource_type=ResourceType.PROCESSOR,
                    message="No active model configuration found for specified project",
                )

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/active")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_active_model_project_not_found(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_active_model(self, project_id):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/active")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_active_model_internal_error(self, client, project_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_active_model(self, project_id):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/active")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetModel:
    def test_get_model_success(self, client, project_id, model_id, sample_processor_schema):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_model(self, project_id, model_id):
                return sample_processor_schema

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == str(model_id)

    def test_get_model_not_found(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_model(self, project_id, model_id):
                raise ResourceNotFoundError(resource_type=ResourceType.PROCESSOR, resource_id=str(model_id))

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_model_project_not_found(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_model(self, project_id, model_id):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_model_internal_error(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def get_model(self, project_id, model_id):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.get(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestCreateModel:
    def test_create_model_success(self, client, project_id, model_id, create_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def create_model(self, project_id, create_data):
                return ProcessorSchema(
                    id=model_id,
                    name=create_data.name,
                    active=False,
                    config=create_data.config,
                )

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.post(f"/api/v1/projects/{project_id}/models", json=create_payload)

        assert response.status_code == status.HTTP_201_CREATED
        assert "Location" in response.headers
        assert response.json()["name"] == "New Model"

    def test_create_model_duplicate_name(self, client, project_id, create_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def create_model(self, project_id, create_data):
                raise ResourceAlreadyExistsError(
                    resource_type=ResourceType.PROCESSOR,
                    resource_value=create_data.name,
                    field="name",
                    message="A model configuration with this name already exists in the project.",
                )

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.post(f"/api/v1/projects/{project_id}/models", json=create_payload)

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in response.json()["detail"].lower()

    def test_create_model_project_not_found(self, client, project_id, create_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def create_model(self, project_id, create_data):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.post(f"/api/v1/projects/{project_id}/models", json=create_payload)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_create_model_internal_error(self, client, project_id, create_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def create_model(self, project_id, create_data):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.post(f"/api/v1/projects/{project_id}/models", json=create_payload)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestUpdateModel:
    def test_update_model_name_success(self, client, project_id, model_id, sample_processor_schema, update_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model(self, project_id, model_id, update_data):
                schema = sample_processor_schema.model_copy()
                schema.name = update_data.name
                return schema

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["name"] == "Update Model"

    def test_update_model_config_success(self, client, project_id, model_id, sample_processor_schema, update_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model(self, project_id, model_id, update_data):
                schema = sample_processor_schema.model_copy()
                if "config" in update_data:
                    schema.config.update(update_data["config"])
                return schema

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_200_OK

    def test_update_model_empty_payload(self, client, project_id, model_id, sample_processor_schema):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model(self, project_id, model_id, update_data):
                return sample_processor_schema

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        payload = {}
        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=payload)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_update_model_not_found(self, client, project_id, model_id, update_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model(self, project_id, model_id, update_data):
                raise ResourceNotFoundError(resource_type=ResourceType.PROCESSOR, resource_id=str(model_id))

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_model_project_not_found(self, client, project_id, model_id, update_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model(self, project_id, model_id, update_data):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_model_internal_error(self, client, project_id, model_id, update_payload):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def update_model(self, project_id, model_id, update_data):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.put(f"/api/v1/projects/{project_id}/models/{model_id}", json=update_payload)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestDeleteModel:
    def test_delete_model_success(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def delete_model(self, project_id, model_id):
                pass

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_model_not_found(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def delete_model(self, project_id, model_id):
                raise ResourceNotFoundError(resource_type=ResourceType.PROCESSOR, resource_id=str(model_id))

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_model_project_not_found(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def delete_model(self, project_id, model_id):
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_model_internal_error(self, client, project_id, model_id):
        class FakeProcessorService:
            def __init__(self, session):
                pass

            def delete_model(self, project_id, model_id):
                raise RuntimeError("Database error")

        client.app.dependency_overrides[get_model_service] = lambda: FakeProcessorService(None)

        response = client.delete(f"/api/v1/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
