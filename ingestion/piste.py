"""PISTE (Légifrance) OAuth2 client with rate limiting and retry.

Endpoints used:
- /consult/getSectionByCid   → section content (LEGISCTA within a code)
- /consult/lawDecree         → whole LODA text (décret, ordonnance)
- /consult/jorf              → raw Journal Officiel text (non-consolidated)
- /consult/getArticle        → single article by LEGIARTI id
"""
from __future__ import annotations
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt,
    wait_exponential, before_sleep_log,
)
import logging

log = logging.getLogger(__name__)

PROD = {
    "token_url": "https://oauth.piste.gouv.fr/api/oauth/token",
    "api_base": "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app",
}
SANDBOX = {
    "token_url": "https://sandbox-oauth.piste.gouv.fr/api/oauth/token",
    "api_base": "https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app",
}


class RateLimiter:
    """Simple thread-safe req/sec limiter."""
    def __init__(self, per_second: float = 5.0):
        self.min_interval = 1.0 / per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()


@dataclass
class PisteClient:
    client_id: str
    client_secret: str
    env: str = "production"
    rate_per_sec: float = 5.0
    timeout_s: float = 30.0

    _token: str | None = field(default=None, init=False)
    _token_expires_at: float = field(default=0.0, init=False)
    _limiter: RateLimiter = field(init=False)
    _http: httpx.Client = field(init=False)

    def __post_init__(self):
        cfg = PROD if self.env == "production" else SANDBOX
        self.token_url = cfg["token_url"]
        self.api_base = cfg["api_base"]
        self._limiter = RateLimiter(self.rate_per_sec)
        self._http = httpx.Client(timeout=self.timeout_s)

    @classmethod
    def from_env(cls) -> "PisteClient":
        env = os.getenv("PISTE_ENV", "sandbox")
        if env == "sandbox":
            cid = os.environ["PISTE_SANDBOX_CLIENT_ID"]
            secret = os.environ["PISTE_SANDBOX_CLIENT_SECRET"]
        else:
            cid = os.environ["PISTE_CLIENT_ID"]
            secret = os.environ["PISTE_CLIENT_SECRET"]
        return cls(client_id=cid, client_secret=secret, env=env)

    def _refresh_token(self) -> None:
        r = self._http.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "openid",
            },
        )
        r.raise_for_status()
        body = r.json()
        self._token = body["access_token"]
        self._token_expires_at = time.monotonic() + body["expires_in"] - 60

    def _auth_headers(self) -> dict[str, str]:
        if self._token is None or time.monotonic() >= self._token_expires_at:
            self._refresh_token()
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._limiter.wait()
        r = self._http.post(
            f"{self.api_base}{path}",
            headers=self._auth_headers(),
            json=payload,
        )
        if r.status_code == 401:
            self._token = None
            r = self._http.post(
                f"{self.api_base}{path}",
                headers=self._auth_headers(),
                json=payload,
            )
        r.raise_for_status()
        return r.json()

    def get_article(self, legiarti_id: str) -> dict[str, Any]:
        return self.post("/consult/getArticle", {"id": legiarti_id})

    def list_articles_in_section(self, section_id: str, parent_text_id: str) -> list[str]:
        payload = {
            "cid": section_id,
            "textCid": parent_text_id,
            "date": _today_ms(),
        }
        data = self.post("/consult/getSectionByCid", payload)
        ids: list[str] = []
        _collect_article_ids(data, ids)
        return ids

    def list_articles_in_loda(self, text_id: str) -> list[str]:
        payload = {"textId": text_id, "date": _today_ms()}
        data = self.post("/consult/lawDecree", payload)
        ids: list[str] = []
        _collect_article_ids(data, ids)
        return ids

    def list_articles_in_jorf(self, text_id: str) -> list[str]:
        payload = {"textCid": text_id, "date": _today_ms()}
        data = self.post("/consult/jorf", payload)
        ids: list[str] = []
        _collect_article_ids(data, ids)
        return ids


def _today_ms() -> int:
    return int(time.time() * 1000)


def _collect_article_ids(node, acc: list[str]) -> None:
    if isinstance(node, dict):
        aid = node.get("id") or node.get("cid")
        if isinstance(aid, str) and aid.startswith("LEGIARTI"):
            acc.append(aid)
        for value in node.values():
            _collect_article_ids(value, acc)
    elif isinstance(node, list):
        for item in node:
            _collect_article_ids(item, acc)
