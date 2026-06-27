import asyncio
import time
import re
import ipaddress
from typing import Optional
from urllib.parse import quote, urlparse
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from cachetools import TTLCache

app = FastAPI()

# ─── Cache (per-instance, tidak persisten antar Vercel cold start) ────────────
cache: TTLCache = TTLCache(maxsize=100, ttl=300)

VALID_SOURCES = {"wikipedia", "duckduckgo", "url_scan", "all"}

# ─── Models ───────────────────────────────────────────────────────────────────
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source: str

class ScannedURL(BaseModel):
    url: str
    status_code: int
    source: str = "url_scan"

class SearchResponse(BaseModel):
    query: str
    total_results: int
    count: int
    results: list[SearchResult]
    scanned_urls: list[ScannedURL]
    timing_ms: int

# ─── Helpers ──────────────────────────────────────────────────────────────────
def clean_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r"[^\w\s\-\.\+\#\/]", "", query)
    return query[:100]

def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname or ""
        if not host:
            return False
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            pass
        blocked_hosts = {"localhost", "metadata.google.internal", "169.254.169.254"}
        if host in blocked_hosts:
            return False
        if host.endswith((".local", ".internal", ".localhost")):
            return False
        return True
    except Exception:
        return False

def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    seen = set()
    out = []
    for r in results:
        key = (
            r.url.rstrip("/")
            .lower()
            .removeprefix("https://")
            .removeprefix("http://")
            .removeprefix("www.")
        )
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

# ─── Wikipedia Adapter ────────────────────────────────────────────────────────
async def fetch_wikipedia(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    results = []
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "format": "json",
            "utf8": 1,
        }
        r = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params=params,
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            encoded_title = quote(title.replace(" ", "_"), safe="_():")
            page_url = f"https://en.wikipedia.org/wiki/{encoded_title}"
            results.append(SearchResult(
                title=title,
                url=page_url,
                snippet=snippet,
                source="wikipedia",
            ))
    except Exception:
        pass
    return results

# ─── DuckDuckGo Adapter ───────────────────────────────────────────────────────
async def fetch_duckduckgo(query: str, client: httpx.AsyncClient) -> list[SearchResult]:
    results = []
    try:
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        r = await client.get(
            "https://api.duckduckgo.com/",
            params=params,
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()

        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append(SearchResult(
                title=data.get("Heading", query),
                url=data["AbstractURL"],
                snippet=data["AbstractText"][:300],
                source="duckduckgo",
            ))

        for topic in data.get("RelatedTopics", [])[:6]:
            if isinstance(topic, dict) and topic.get("FirstURL") and topic.get("Text"):
                results.append(SearchResult(
                    title=topic["Text"][:80],
                    url=topic["FirstURL"],
                    snippet=topic["Text"][:200],
                    source="duckduckgo",
                ))

        for result in data.get("Results", [])[:3]:
            if result.get("FirstURL") and result.get("Text"):
                results.append(SearchResult(
                    title=result["Text"][:80],
                    url=result["FirstURL"],
                    snippet=result["Text"][:200],
                    source="duckduckgo",
                ))
    except Exception:
        pass
    return results

# ─── URL Scanner ──────────────────────────────────────────────────────────────
def generate_candidate_urls(query: str) -> list[str]:
    slug = re.sub(r"\s+", "", query.lower())
    slug_dash = re.sub(r"\s+", "-", query.lower().strip())
    slug_dot = re.sub(r"\s+", ".", query.lower().strip())
    words = query.lower().strip().split()
    first = words[0] if words else slug

    candidates = [
        # Root domains
        f"https://{slug}.org",
        f"https://{slug}.com",
        f"https://{slug}.io",
        f"https://{slug}.dev",
        f"https://{slug}.net",
        f"https://{slug}.co",
        f"https://{slug_dash}.org",
        f"https://{slug_dash}.com",
        f"https://{slug_dash}.io",
        f"https://{slug_dash}.dev",
        # WWW variants
        f"https://www.{slug}.org",
        f"https://www.{slug}.com",
        f"https://www.{slug}.net",
        # Docs subdomains
        f"https://docs.{slug}.org",
        f"https://docs.{slug}.com",
        f"https://docs.{slug}.io",
        f"https://docs.{slug_dash}.io",
        f"https://docs.{slug_dash}.org",
        # Read the Docs
        f"https://{slug}.readthedocs.io",
        f"https://{slug_dash}.readthedocs.io",
        # PyPI
        f"https://pypi.org/project/{slug}/",
        f"https://pypi.org/project/{slug_dash}/",
        # npm
        f"https://www.npmjs.com/package/{slug}",
        f"https://www.npmjs.com/package/{slug_dash}",
        f"https://www.npmjs.com/package/@{slug}/{slug}",
        # Crates.io
        f"https://crates.io/crates/{slug}",
        f"https://crates.io/crates/{slug_dash}",
        # Go packages
        f"https://pkg.go.dev/{slug}",
        f"https://pkg.go.dev/{slug_dot}",
        # GitHub
        f"https://github.com/{slug}/{slug}",
        f"https://github.com/topics/{slug}",
        f"https://github.com/topics/{slug_dash}",
        # Wikipedia
        f"https://en.wikipedia.org/wiki/{quote(query.replace(' ', '_'), safe='_():')}",
        # MDN
        f"https://developer.mozilla.org/en-US/docs/Web/API/{slug}",
        f"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/{slug}",
        # StackOverflow
        f"https://stackoverflow.com/questions/tagged/{slug}",
        f"https://stackoverflow.com/questions/tagged/{slug_dash}",
        # Awesome lists
        f"https://github.com/topics/awesome-{slug}",
        f"https://github.com/topics/awesome-{slug_dash}",
        # Homebrew
        f"https://formulae.brew.sh/formula/{slug}",
        f"https://formulae.brew.sh/cask/{slug}",
        # Docker Hub
        f"https://hub.docker.com/_/{slug}",
        f"https://hub.docker.com/r/{slug}/{slug}",
        # Ruby Gems
        f"https://rubygems.org/gems/{slug}",
        f"https://rubygems.org/gems/{slug_dash}",
        # NuGet
        f"https://www.nuget.org/packages/{slug}/",
        # Maven
        f"https://mvnrepository.com/artifact/{slug}",
        f"https://{slug}.org",
        f"https://{slug}.com",
        f"https://{slug}.io",
        f"https://{slug}.dev",
        f"https://{slug}.net",
        f"https://{slug}.co",
        f"https://{slug_dash}.org",
        f"https://{slug_dash}.com",
        f"https://{slug_dash}.io",
        f"https://{slug_dash}.dev",
        # WWW variants
        f"https://www.{slug}.org",
        f"https://www.{slug}.com",
        f"https://www.{slug}.net",
        # Docs subdomains
        f"https://docs.{slug}.org",
        f"https://docs.{slug}.com",
        f"https://docs.{slug}.io",
        f"https://docs.{slug_dash}.io",
        f"https://docs.{slug_dash}.org",
        # Read the Docs
        f"https://{slug}.readthedocs.io",
        f"https://{slug_dash}.readthedocs.io",
        # PyPI
        f"https://pypi.org/project/{slug}/",
        f"https://pypi.org/project/{slug_dash}/",
        # npm
        f"https://www.npmjs.com/package/{slug}",
        f"https://www.npmjs.com/package/{slug_dash}",
        f"https://www.npmjs.com/package/@{slug}/{slug}",
        # Crates.io
        f"https://crates.io/crates/{slug}",
        f"https://crates.io/crates/{slug_dash}",
        # Go packages
        f"https://pkg.go.dev/{slug}",
        f"https://pkg.go.dev/{slug_dot}",
        # GitHub
        f"https://github.com/{slug}/{slug}",
        f"https://github.com/topics/{slug}",
        f"https://github.com/topics/{slug_dash}",
        # Wikipedia
        f"https://en.wikipedia.org/wiki/{quote(query.replace(' ', '_'), safe='_():')}",
        # MDN
        f"https://developer.mozilla.org/en-US/docs/Web/API/{slug}",
        f"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/{slug}",
        # StackOverflow
        f"https://stackoverflow.com/questions/tagged/{slug}",
        f"https://stackoverflow.com/questions/tagged/{slug_dash}",
        # Awesome lists
        f"https://github.com/topics/awesome-{slug}",
        f"https://github.com/topics/awesome-{slug_dash}",
        # Homebrew
        f"https://formulae.brew.sh/formula/{slug}",
        f"https://formulae.brew.sh/cask/{slug}",
        # Docker Hub
        f"https://hub.docker.com/_/{slug}",
        f"https://hub.docker.com/r/{slug}/{slug}",
        # Ruby Gems
        f"https://rubygems.org/gems/{slug}",
        f"https://rubygems.org/gems/{slug_dash}",
        # NuGet
        f"https://www.nuget.org/packages/{slug}/",
        # Maven
        f"https://mvnrepository.com/artifact/{slug}",
    ]

    # Multi-word first-word variants
    if len(words) > 1:
        candidates += [
            f"https://{first}.org",
            f"https://{first}.com",
            f"https://{first}.io",
            f"https://docs.{first}.org",
            f"https://docs.{first}.io",
        ]

    # Deduplicate, filter safe, cap at 50
    seen: set[str] = set()
    unique: list[str] = []
    for u in candidates:
        if u not in seen and is_safe_url(u):
            seen.add(u)
            unique.append(u)
    return unique[:50]

async def scan_urls(query: str, client: httpx.AsyncClient) -> list[ScannedURL]:
    candidates = generate_candidate_urls(query)
    results: list[ScannedURL] = []

    async def check(url: str) -> Optional[ScannedURL]:
        try:
            r = await client.head(url, timeout=4, follow_redirects=True)
            if r.status_code == 200:
                return ScannedURL(url=url, status_code=200)
            # Beberapa server reject HEAD, fallback ke GET
            if r.status_code in (405, 403):
                r2 = await client.get(url, timeout=4, follow_redirects=True)
                if r2.status_code == 200:
                    return ScannedURL(url=url, status_code=200)
        except Exception:
            pass
        return None

    raw = await asyncio.gather(*[check(u) for u in candidates])
    for item in raw:
        if item is not None:
            results.append(item)
    return results

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(content={
        "service": "SearchAI",
        "version": "2.0.0",
        "endpoint": "/api/search",
        "params": {
            "query": "required | string | max 100 chars",
            "limit": "optional | int 1–50 | default 15",
            "source": "optional | wikipedia | duckduckgo | url_scan | all",
        },
        "note": "Cache is per-instance only. Not persistent across Vercel cold starts.",
    })

@app.get("/api/search")
async def search(
    query: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(default=15, ge=1, le=50),
    source: Optional[str] = Query(default=None),
) -> JSONResponse:
    # Validasi source
    if source is not None and source not in VALID_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source '{source}'. Valid values: {', '.join(sorted(VALID_SOURCES))}",
        )
    if source == "all":
        source = None

    q = clean_query(query)
    if not q:
        raise HTTPException(status_code=400, detail="Query is empty after cleaning.")

    cache_key = f"{q}:{limit}:{source}"
    if cache_key in cache:
        response = cache[cache_key]
        return JSONResponse(content=response, headers={"X-Cache": "HIT"})

    t0 = time.monotonic()

    async def empty() -> list:
        return []

    async with httpx.AsyncClient(
        headers={"User-Agent": "SearchAI/2.0 (+https://github.com/searchai)"},
        follow_redirects=True,
    ) as client:
        wiki_task   = fetch_wikipedia(q, client)  if source in (None, "wikipedia")  else empty()
        ddg_task    = fetch_duckduckgo(q, client) if source in (None, "duckduckgo") else empty()
        scan_task   = scan_urls(q, client)        if source in (None, "url_scan")   else empty()

        wiki_results, ddg_results, scanned = await asyncio.gather(
            wiki_task, ddg_task, scan_task
        )

    merged_all = deduplicate(wiki_results + ddg_results)
    total = len(merged_all)
    merged = merged_all[:limit]

    timing = int((time.monotonic() - t0) * 1000)

    response = SearchResponse(
        query=q,
        total_results=total,
        count=len(merged),
        results=merged,
        scanned_urls=scanned,
        timing_ms=timing,
    ).model_dump()

    cache[cache_key] = response
    return JSONResponse(content=response, headers={"X-Cache": "MISS"})
