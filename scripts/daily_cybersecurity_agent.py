#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import requests

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


RSS_FEEDS = {
    "BleepingComputer": "https://www.bleepingcomputer.com/feed/",
    "The Hacker News": "https://feeds.feedburner.com/TheHackersNews",
    "SecurityWeek": "https://www.securityweek.com/feed/",
    "Dark Reading": "https://www.darkreading.com/rss.xml",
    "Krebs on Security": "https://krebsonsecurity.com/feed/",
}
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


@dataclass
class FeedItem:
    source: str
    title: str
    link: str
    published: datetime


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def fetch_rss_items(days: int) -> list[FeedItem]:
    cutoff = utc_now() - timedelta(days=days)
    items: list[FeedItem] = []
    for source, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            logging.warning("Failed to fetch/parse feed %s (%s): %s", source, url, exc)
            continue

        # RSS
        for node in root.findall(".//item"):
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            published = parse_date(node.findtext("pubDate"))
            if title and link and published and published >= cutoff:
                items.append(FeedItem(source=source, title=title, link=link, published=published))

        # Atom
        atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        for node in atom_entries:
            title = (node.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
            link_node = node.find("{http://www.w3.org/2005/Atom}link")
            link = (link_node.get("href") if link_node is not None else "") or ""
            published = parse_date(
                node.findtext("{http://www.w3.org/2005/Atom}updated")
                or node.findtext("{http://www.w3.org/2005/Atom}published")
            )
            if title and link and published and published >= cutoff:
                items.append(FeedItem(source=source, title=title, link=link, published=published))

    items.sort(key=lambda x: x.published, reverse=True)
    return items[:40]


def fetch_recent_kev(days: int) -> list[dict]:
    cutoff = utc_now() - timedelta(days=days)
    try:
        resp = requests.get(KEV_URL, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logging.warning("Failed to fetch/parse CISA KEV feed: %s", exc)
        return []

    recent: list[dict] = []
    for vuln in payload.get("vulnerabilities", []):
        raw = vuln.get("dateAdded")
        if not raw:
            continue
        try:
            added = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if added >= cutoff:
            recent.append(
                {
                    "cve": vuln.get("cveID", ""),
                    "vendorProject": vuln.get("vendorProject", ""),
                    "product": vuln.get("product", ""),
                    "dateAdded": raw,
                    "requiredAction": vuln.get("requiredAction", ""),
                    "notes": vuln.get("notes", ""),
                }
            )
    return recent[:30]


def build_prompt(feed_items: Iterable[FeedItem], kev_items: Iterable[dict], days: int) -> str:
    feed_payload = [
        {
            "source": f.source,
            "title": f.title,
            "link": f.link,
            "published": f.published.isoformat(),
        }
        for f in feed_items
    ]
    return f"""
You are a Senior Cybersecurity Research Analyst.
Create an executive daily cybersecurity summary using only the provided data from the last {days} day(s).
Do not invent facts.

Required sections:
1) Executive Summary (top developments)
2) New/Notable KEV Entries (if available)
3) Emerging Threats and Attack Trends
4) Incidents and Breaches
5) Immediate Actions (0-24h), Short Term (1-7d), Medium Term (7-30d)
6) Risk Dashboard table (Critical/High/Medium/Low counts based on evidence in provided data)

Include source references with publication dates and links for each major finding.
Use concise, executive-friendly markdown.

News data:
{json.dumps(feed_payload, indent=2)}

KEV data:
{json.dumps(list(kev_items), indent=2)}
""".strip()


def generate_with_llm(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        return content.strip()
    except Exception as exc:
        logging.warning("LLM generation failed, using fallback report: %s", exc)
        return None


def fallback_report(feed_items: list[FeedItem], kev_items: list[dict], days: int) -> str:
    lines = [
        f"# Daily Cybersecurity Summary ({utc_now().date().isoformat()})",
        "",
        f"Coverage window: last {days} day(s)",
        "",
        "## Executive Summary",
        f"- Collected {len(feed_items)} security news items from configured feeds.",
        f"- Found {len(kev_items)} CISA KEV entries added in the same window.",
        "",
        "## Recent Security News",
    ]
    if feed_items:
        for item in feed_items[:15]:
            lines.append(
                f"- [{item.title}]({item.link}) — {item.source} ({item.published.date().isoformat()})"
            )
    else:
        lines.append("- No recent news items collected from configured feeds.")

    lines.extend(["", "## New/Notable KEV Entries"])
    if kev_items:
        for item in kev_items:
            lines.append(
                f"- **{item['cve']}** | {item['vendorProject']} {item['product']} | Added: {item['dateAdded']} | Action: {item['requiredAction']}"
            )
    else:
        lines.append("- No new KEV entries in this window.")

    lines.extend(
        [
            "",
            "## Immediate Actions (0-24h)",
            "- Patch any KEV-listed internet-facing assets first.",
            "- Verify EDR/SIEM detection coverage for impacted products.",
            "",
            "## Short Term (1-7d)",
            "- Complete patch rollout for high-risk systems and validate mitigations.",
            "",
            "## Medium Term (7-30d)",
            "- Run exposure review and close gaps in vulnerability management SLAs.",
            "",
            "## Sources",
        ]
    )
    for name, url in RSS_FEEDS.items():
        lines.append(f"- {name}: {url}")
    lines.append(f"- CISA KEV: {KEV_URL}")
    return "\n".join(lines)


def write_report(content: str) -> Path:
    output_dir = Path(os.getenv("OUTPUT_DIR", "reports/daily"))
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"cybersecurity-summary-{utc_now().date().isoformat()}.md"
    output_path = output_dir / filename
    output_path.write_text(content + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    days = int(os.getenv("LOOKBACK_DAYS", "1"))
    feed_items = fetch_rss_items(days)
    kev_items = fetch_recent_kev(days)
    prompt = build_prompt(feed_items, kev_items, days)

    report = generate_with_llm(prompt)
    if not report:
        report = fallback_report(feed_items, kev_items, days)

    output = write_report(report)
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()
