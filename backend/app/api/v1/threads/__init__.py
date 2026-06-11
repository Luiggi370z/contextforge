"""Threads entity: HTTP router, service, repository, schemas, and ORM model.

This package ``__init__`` is intentionally empty: importing the ORM (``.model``)
from core code must not pull in the router/HTTP stack. Import the router from
``app.api.v1.threads.router`` directly.
"""
