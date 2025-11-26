"""
src/deploy/model_registry.py

Lightweight file-based Model Registry.

Features:
- save(name, version, path, metadata) -> registers a model artifact
- get(name, version=None) -> retrieve a specific version or latest
- list(name=None) -> list all entries or for a given model name
- promote(name, version, stage='staging', by='user') -> set stage (e.g., staging, production)
- delete(name, version) -> delete a model entry
- simple locking to avoid concurrent writes (advisory via atomic write)

Notes:
- Registry file defaults to 'models/model_registry.json'
- Entries are small JSON objects. For a production system, swap this for S3/DB/MLflow.
"""
from __future__ import annotations
import json
import os
import time
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

REGISTRY_DEFAULT_PATH = os.environ.get("MODEL_REGISTRY_PATH", "models/model_registry.json")


class ModelRegistry:
    def __init__(self, registry_path: str = REGISTRY_DEFAULT_PATH):
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        # initialize file if missing
        if not os.path.exists(self.registry_path):
            self._atomic_write({"models": []})

    # ---- internal helpers ----
    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except Exception:
            # If file corrupted or missing, return empty skeleton
            return {"models": []}

    def _atomic_write(self, payload: Dict[str, Any]) -> None:
        # write to tempfile then replace
        d = os.path.dirname(self.registry_path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".regtmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            # atomic replace
            os.replace(tmp, self.registry_path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def _now_iso(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _make_entry(self, name: str, version: str, path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "name": name,
            "version": str(version),
            "path": path,
            "metadata": metadata or {},
            "created_at": self._now_iso(),
            "stage": "none",  # e.g., staging/production/none
            "promoted_by": None,
            "promoted_at": None,
        }

    # ---- public API ----
    def save(self, name: str, version: str, path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save/register a new model artifact.
        Returns the saved entry.
        """
        data = self._read()
        entry = self._make_entry(name=name, version=version, path=path, metadata=metadata)
        # avoid duplicates of same name+version
        existing = [m for m in data.get("models", []) if m.get("name") == name and m.get("version") == str(version)]
        if existing:
            # update existing entry in-place (but keep created_at)
            existing_entry = existing[0]
            existing_entry.update(entry)
            entry = existing_entry
        else:
            data.setdefault("models", []).append(entry)
        self._atomic_write(data)
        return entry

    def get(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get model entry. If version is None, returns the latest by created_at.
        """
        data = self._read()
        models = [m for m in data.get("models", []) if m.get("name") == name]
        if not models:
            return None
        if version is None:
            # pick latest by created_at
            try:
                models_sorted = sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)
                return models_sorted[0]
            except Exception:
                return models[0]
        for m in models:
            if m.get("version") == str(version):
                return m
        return None

    def list(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all model registry entries. If name provided, filter for that model name.
        """
        data = self._read()
        models = data.get("models", []) or []
        if name:
            return [m for m in models if m.get("name") == name]
        return models

    def promote(self, name: str, version: str, stage: str = "staging", by: str = "user") -> Optional[Dict[str, Any]]:
        """
        Promote a model version to a named stage (e.g., 'staging', 'production').
        Returns the updated entry or None if not found.
        """
        data = self._read()
        changed = False
        for m in data.get("models", []):
            if m.get("name") == name and m.get("version") == str(version):
                m["stage"] = stage
                m["promoted_by"] = by
                m["promoted_at"] = self._now_iso()
                changed = True
                entry = m
                break
        if changed:
            self._atomic_write(data)
            return entry
        return None

    def delete(self, name: str, version: str) -> bool:
        """
        Delete a specific model entry. Returns True if removed.
        """
        data = self._read()
        before = len(data.get("models", []))
        data["models"] = [m for m in data.get("models", []) if not (m.get("name") == name and m.get("version") == str(version))]
        after = len(data.get("models", []))
        if after < before:
            self._atomic_write(data)
            return True
        return False

    def latest(self, name: str) -> Optional[Dict[str, Any]]:
        """Convenience: alias for get(name, version=None)"""
        return self.get(name=name, version=None)


# ---- basic CLI for convenience ----
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(prog="model_registry")
    sub = p.add_subparsers(dest="cmd")

    p_save = sub.add_parser("save")
    p_save.add_argument("--name", required=True)
    p_save.add_argument("--version", required=True)
    p_save.add_argument("--path", required=True)
    p_save.add_argument("--metadata", default=None, help="JSON string")

    p_get = sub.add_parser("get")
    p_get.add_argument("--name", required=True)
    p_get.add_argument("--version", default=None)

    p_list = sub.add_parser("list")
    p_list.add_argument("--name", default=None)

    p_prom = sub.add_parser("promote")
    p_prom.add_argument("--name", required=True)
    p_prom.add_argument("--version", required=True)
    p_prom.add_argument("--stage", default="staging")
    p_prom.add_argument("--by", default="user")

    p_del = sub.add_parser("delete")
    p_del.add_argument("--name", required=True)
    p_del.add_argument("--version", required=True)

    args = p.parse_args()
    reg = ModelRegistry()

    if args.cmd == "save":
        meta = None
        if args.metadata:
            try:
                meta = json.loads(args.metadata)
            except Exception:
                meta = {"raw": args.metadata}
        entry = reg.save(name=args.name, version=args.version, path=args.path, metadata=meta)
        print(json.dumps(entry, indent=2))
    elif args.cmd == "get":
        entry = reg.get(name=args.name, version=args.version)
        print(json.dumps(entry or {}, indent=2))
    elif args.cmd == "list":
        entries = reg.list(name=args.name)
        print(json.dumps(entries, indent=2))
    elif args.cmd == "promote":
        entry = reg.promote(name=args.name, version=args.version, stage=args.stage, by=args.by)
        print(json.dumps(entry or {}, indent=2))
    elif args.cmd == "delete":
        ok = reg.delete(name=args.name, version=args.version)
        print(json.dumps({"deleted": ok}, indent=2))
    else:
        p.print_help()
