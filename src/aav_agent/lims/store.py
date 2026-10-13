"""Run and screen records. One interface, an in-memory store for tests/dev, and MongoDB."""

from __future__ import annotations

import copy
from typing import Protocol


class RunStore(Protocol):
    def create_run(self, doc: dict) -> None: ...
    def update_run(self, run_id: str, fields: dict) -> None: ...
    def get_run(self, run_id: str) -> dict | None: ...
    def list_runs(self, limit: int = 20) -> list[dict]: ...
    def create_screen(self, doc: dict) -> None: ...
    def update_screen(self, doc_id: str, fields: dict) -> None: ...
    def list_screens(self, run_id: str) -> list[dict]: ...


class InMemoryRunStore:
    """Dictionary-backed store. Documents are copied on the way in and out, like a real database."""

    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}
        self._screens: dict[str, dict] = {}

    def create_run(self, doc: dict) -> None:
        if doc["_id"] in self._runs:
            raise ValueError(f"run {doc['_id']} already exists")
        self._runs[doc["_id"]] = copy.deepcopy(doc)

    def update_run(self, run_id: str, fields: dict) -> None:
        self._runs[run_id].update(copy.deepcopy(fields))

    def get_run(self, run_id: str) -> dict | None:
        d = self._runs.get(run_id)
        return copy.deepcopy(d) if d else None

    def list_runs(self, limit: int = 20) -> list[dict]:
        docs = sorted(self._runs.values(), key=lambda d: d["created_utc"], reverse=True)
        return [copy.deepcopy(d) for d in docs[:limit]]

    def create_screen(self, doc: dict) -> None:
        self._screens[doc["_id"]] = copy.deepcopy(doc)

    def update_screen(self, doc_id: str, fields: dict) -> None:
        self._screens[doc_id].update(copy.deepcopy(fields))

    def list_screens(self, run_id: str) -> list[dict]:
        docs = sorted((d for d in self._screens.values() if d["run_id"] == run_id),
                      key=lambda d: d["created_utc"])
        return [copy.deepcopy(d) for d in docs]


class MongoRunStore:
    """MongoDB-backed store. `db` is a pymongo Database (or anything with the same collection API).

    Timestamps are stored as ISO-8601 UTC strings, so they sort correctly and serialise to JSON
    without conversion.
    """

    def __init__(self, db) -> None:
        self.runs = db["runs"]
        self.screens = db["screens"]

    @classmethod
    def from_uri(cls, uri: str, db_name: str = "aav_agent") -> MongoRunStore:
        import pymongo  # imported here so the rest of the package works without it

        return cls(pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)[db_name])

    def create_run(self, doc: dict) -> None:
        self.runs.insert_one(copy.deepcopy(doc))

    def update_run(self, run_id: str, fields: dict) -> None:
        self.runs.update_one({"_id": run_id}, {"$set": fields})

    def get_run(self, run_id: str) -> dict | None:
        return self.runs.find_one({"_id": run_id})

    def list_runs(self, limit: int = 20) -> list[dict]:
        return list(self.runs.find({}).sort("created_utc", -1).limit(limit))

    def create_screen(self, doc: dict) -> None:
        self.screens.insert_one(copy.deepcopy(doc))

    def update_screen(self, doc_id: str, fields: dict) -> None:
        self.screens.update_one({"_id": doc_id}, {"$set": fields})

    def list_screens(self, run_id: str) -> list[dict]:
        return list(self.screens.find({"run_id": run_id}).sort("created_utc", 1))
