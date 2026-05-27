"""Pydantic bases for HTTP JSON (snake_case Python ↔ camelCase JSON)."""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class WithAliasModel(BaseModel):
    """Convert snake_case field names to camelCase for JSON serialization."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


class BaseRequest(WithAliasModel):
    """Request bodies from the frontend.

    Accepts camelCase (JS clients) and snake_case (internal/tests).
    """


class BaseResponse(WithAliasModel):
    """Response bodies sent to the frontend.

    Serializes snake_case Python fields as camelCase JSON keys.
    """
