"""PISTE (Légifrance) OAuth2 client with rate limiting and retry.

Endpoints used:
- /consult/getSectionByCid   → section content (LEGISCTA within a code)
- /consult/lawDecree         → whole LODA text (décret, ordonnance)
- /consult/jorf              → raw Journal Officiel text (non-consolidated)
- /consult/getArticle        → single article by LEGIARTI id
"""
from __future__ import annotations
import os
import re
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

_VALID_ARTI = re.compile(r"^LEGIARTI\d{12}$")
_VALID_SCTA = re.compile(r"^LEGISCTA\d{12}$")

PROD = {
    "token_url": "https://oauth.piste.gouv.fr/api/oauth/token",
    "api_base": "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app",
}
SANDBOX = {
    "token_url": "https://sandbox-oauth.piste.gouv.fr/api/oauth/token",
    "api_base": "https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app",
}


class RateLimiter:
    """Simple rate limiter to keep API requests below a fixed rate."""

    def __init__(self, per_second: float = 5.0):
        self.min_interval = 1.0 / per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until the next allowed request time has arrived."""
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()


@dataclass
class PisteClient:
    """Client for PISTE OAuth2 authentication and Legifrance data calls."""

    client_id: str
    client_secret: str
    env: str = "sandbox"
    rate_per_sec: float = 3.0
    timeout_s: float = 30.0

    _token: str | None = field(default=None, init=False)
    _token_expires_at: float = field(default=0.0, init=False)
    _limiter: RateLimiter = field(init=False)
    _http: httpx.Client = field(init=False)

    def __post_init__(self):
        """Set endpoint URLs, rate limiter, and HTTP client after object creation."""
        cfg = PROD if self.env == "production" else SANDBOX
        self.token_url = cfg["token_url"]
        self.api_base = cfg["api_base"]
        self._limiter = RateLimiter(self.rate_per_sec)
        self._http = httpx.Client(timeout=self.timeout_s)

    @classmethod
    def from_env(cls) -> "PisteClient":
        """Create a client using credentials read from environment variables."""
        env = os.getenv("PISTE_ENV", "sandbox")
        if env == "sandbox":
            cid = os.environ["PISTE_SANDBOX_CLIENT_ID"]
            secret = os.environ["PISTE_SANDBOX_CLIENT_SECRET"]
        else:
            cid = os.environ["PISTE_CLIENT_ID"]
            secret = os.environ["PISTE_CLIENT_SECRET"]
        return cls(client_id=cid, client_secret=secret, env=env)

    def _refresh_token(self) -> None:
        """Fetch a new OAuth2 access token and cache its expiry time."""
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
        """Return bearer headers, refreshing token first if expired."""
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
        """Post JSON to a PISTE endpoint, with auth, retry, and rate limiting."""
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
        if r.status_code >= 400:
            log.error("PISTE %s → %d: %s", path, r.status_code, r.text[:500])
        r.raise_for_status()
        return r.json()

    # --------------------------------------------------------- endpoints
    def get_article(self, legiarti_id: str) -> dict[str, Any]:
        """Fetch one article object from PISTE by its LEGIARTI id."""
        return self.post("/consult/getArticle", {"id": legiarti_id})

    def list_articles_in_section(
        self,
        section_id: str,
        parent_text_id: str,
        _depth: int = 0,
        _visited: set[str] | None = None,
    ) -> list[str]:
        """Collect all article ids under a section, following nested subsections."""
        if _visited is None:
            _visited = set()
        if section_id in _visited:
            return []
        _visited.add(section_id)

        if _depth > 8:
            log.warning("max recursion depth on %s", section_id)
            return []

        payload = {
            "cid": section_id,
            "textCid": parent_text_id,
            "date": _today_ms(),
        }
        data = self.post("/consult/getSectionByCid", payload)

        articles: list[str] = []
        child_sections: list[str] = []
        _collect_ids(data, articles, child_sections)

        for sub_id in child_sections:
            if sub_id in _visited:
                continue
            try:
                articles.extend(
                    self.list_articles_in_section(
                        sub_id,
                        parent_text_id,
                        _depth + 1,
                        _visited,
                    )
                )
            except Exception as e:
                log.warning("subsection %s failed: %s", sub_id, e)

        seen = set()
        return [a for a in articles if not (a in seen or seen.add(a))]

    def list_articles_in_loda(self, text_id: str) -> list[str]:
        """Fetch all article ids from a law or decree text by its text identifier."""
        payload = {"textId": text_id, "date": _today_ms()}
        data = self.post("/consult/lawDecree", payload)
        articles: list[str] = []
        sections: list[str] = []
        _collect_ids(data, articles, sections)
        return articles

    def list_articles_in_jorf(self, text_id: str) -> list[str]:
        """Fetch article ids from Journal Officiel raw text, trying alternate request keys."""
        # PISTE /consult/jorf expects "textCid" per Swagger; try alternate keys on 400
        for key in ("textCid", "id", "textId"):
            try:
                data = self.post("/consult/jorf", {key: text_id, "date": _today_ms()})
                articles: list[str] = []
                sections: list[str] = []
                _collect_ids(data, articles, sections)
                if articles:
                    return articles
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 400:
                    raise
        log.warning("all jorf payload variants failed for %s", text_id)
        return []


def _today_ms() -> int:
    """Return the current epoch time in milliseconds for API payloads."""
    return int(time.time() * 1000)


# Keys that embed references to OTHER articles — not children of this section.
_SKIP_KEYS = {
    "articleVersions",     # historical versions of the same article
    "lienCitations",       # articles cited by this one
    "lienModifications",   # articles that modified this one
    "lienConcordes",       # concordance links
    "lienAutres",          # other links
    "versionPrecedente",   # previous version id (bare string)
    "textTitles",          # parent text metadata (may contain unrelated ids)
    "context",             # ANCESTOR chain — walking this drags in grandparents & siblings
    "titreTxt",            # parent text metadata inside context
}


def _collect_ids(node, articles: list[str], sections: list[str]) -> None:
    """Walk any nested structure. Collect LEGIARTI into articles, LEGISCTA into sections.

    Skips reference-embedding keys that point to articles/sections OUTSIDE the
    current tree (citations, historical versions, cross-code links) so we
    don't over-enumerate and don't mislabel foreign articles.
    """
    if isinstance(node, dict):
        aid = node.get("id") or node.get("cid")
        etat = node.get("etat", "")
        if isinstance(aid, str):
            if _VALID_ARTI.match(aid):
                if etat in ("", "VIGUEUR", "VIGUEUR_DIFF"):
                    articles.append(aid)
            elif _VALID_SCTA.match(aid):
                sections.append(aid)
        for k, v in node.items():
            if k in _SKIP_KEYS:
                continue
            _collect_ids(v, articles, sections)
    elif isinstance(node, list):
        for item in node:
            _collect_ids(item, articles, sections)