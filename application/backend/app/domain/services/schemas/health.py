from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class HealthStatus(StrEnum):
    OK = "ok"


class HealthCheckSchema(BaseModel):
    status: Literal[HealthStatus.OK]
