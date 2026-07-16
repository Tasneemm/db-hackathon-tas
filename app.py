import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    from google.cloud import storage
except Exception:  # pragma: no cover - optional dependency
    storage = None

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency
    genai = None

HEADERS = {"User-Agent": "Mozilla/5.0"}


BASE_URL = "https://artificialintelligenceact.eu/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_PREFIX = os.getenv("GCS_PREFIX", "ai-act-data")
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_MODEL = os.getenv("GCP_MODEL", "gemini-2.0-flash-001")


def fetch_homepage() -> str:
    response = requests.get(BASE_URL, timeout=20)
    response.raise_for_status()
    return response.text


def fetch_article_text(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for container in soup.select("article, main, .entry-content, .post-content"):
            text = " ".join(container.get_text(" ", strip=True).split())
            if len(text) > 80:
                return text
        return " ".join(soup.get_text(" ", strip=True).split())
    except Exception:
        return ""


def parse_articles(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, str]] = []

    for article in soup.select("article"):
        title_tag = article.select_one("h2, h3, h4, a")
        if title_tag is None:
            continue

        title = " ".join(title_tag.get_text(" ", strip=True).split())
        link = title_tag.get("href") or ""
        if link and not link.startswith("http"):
            link = urljoin(BASE_URL, link)

        if not link:
            continue

        text = fetch_article_text(link)
        excerpt = " ".join(text.split())
        if len(excerpt) > 500:
            excerpt = excerpt[:500].rstrip() + "..."

        items.append({
            "title": title,
            "date": "",
            "link": link,
            "excerpt": excerpt or title,
        })

    for link in soup.select("a[href]"):
        href = link.get("href", "")
        if not href or not href.startswith("http"):
            continue
        if "artificialintelligenceact.eu" not in href:
            continue
        text = " ".join(link.get_text(" ", strip=True).split())
        if not text:
            continue
        if any(keyword in href.lower() for keyword in ["ai-act", "act", "compliance", "guidance", "sandbox", "gpai", "transparency", "implementation"]):
            if not any(item["link"] == href for item in items):
                content = fetch_article_text(href)
                excerpt = " ".join(content.split())
                if len(excerpt) > 500:
                    excerpt = excerpt[:500].rstrip() + "..."
                items.append({
                    "title": text,
                    "date": "",
                    "link": href,
                    "excerpt": excerpt or text,
                })

    unique: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        if item["link"] and item["link"] not in seen:
            seen.add(item["link"])
            unique.append(item)
    return unique[:20]


def build_gcs_blob_name(kind: str, fetched_at: str) -> str:
    return f"{GCS_PREFIX}/{kind}/{fetched_at}/{'raw_articles.json' if kind == 'raw' else 'llm_output.json'}"


def create_raw_payload(articles: List[Dict[str, str]]) -> Dict[str, Any]:
    payload = {
        "source": BASE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "articles": articles,
    }
    return payload


def upload_to_gcs(payload: Dict[str, Any], kind: str) -> str | None:
    if not GCS_BUCKET or storage is None:
        return None

    client = storage.Client(project=GCP_PROJECT)
    blob_name = build_gcs_blob_name(kind, payload["fetched_at"])
    blob = client.bucket(GCS_BUCKET).blob(blob_name) # type: ignore
    blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")
    return f"gs://{GCS_BUCKET}/{blob_name}"


def build_prompt(articles: List[Dict[str, Any]]) -> str:
    article_text = json.dumps(articles, indent=2)
    return f"""
You are a Senior Banking Regulatory Compliance Officer with expertise in:

- EU AI Act
- DORA
- GDPR
- Basel
- EBA Guidelines

Your job is to analyse regulatory documents and estimate their impact on a typical retail and commercial bank.

--------------------------------------------------

TASK

Read the regulation below.

Identify:

1. Regulation Name
2. Article Number (if available)
3. Effective Date
4. Compliance Deadline
5. Summary (maximum 100 words)
6. Whether the requirement is Mandatory or Advisory
7. Regulatory Authority
8. Applicable Geography

--------------------------------------------------

Determine which banking domains are impacted.

Choose only from:
- Loans
- Deposits
- Payments
- Credit Cards
- AML
- KYC
- Treasury
- Wealth Management
- Digital Banking
- Customer Onboarding
- Fraud Detection
- Third Party Risk
- Cyber Security
- Data Governance
- Risk Management
- Compliance
For each domain provide
Domain Name
Confidence Score (0-100)
Reason
--------------------------------------------------
Evaluate the following risk dimensions.
Give a score between 1 and 10.
Regulatory Severity
Business Impact
Operational Complexity
Technology Impact
Customer Impact
Data Privacy Impact
Cyber Security Impact
Financial Penalty
Implementation Difficulty
Urgency
--------------------------------------------------
Calculate
Overall Risk Score (0-100)
Risk Category
Low
Medium
High
Critical
--------------------------------------------------
Provide reasoning for every score.
--------------------------------------------------
Return JSON only.

    Data:
{article_text}
"""


def normalize_llm_output(raw_output: str) -> Dict[str, Any]:
    text = raw_output.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text)


def generate_llm_json(prompt: str, articles: List[Dict[str, Any]]) -> str:
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and genai is not None and GCP_PROJECT:
        genai.configure(project=GCP_PROJECT, location=GCP_LOCATION)
        model = genai.GenerativeModel(GCP_MODEL)
        response = model.generate_content(prompt)
        return response.text

    if os.getenv("OPENAI_API_KEY") and OpenAI is not None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        return response.output_text

    return json.dumps({
        "source": BASE_URL,
        "items": [
            {
                "title": article.get("title", "Untitled"),
                "date": article.get("date") or "Unknown",
                "url": article.get("link", ""),
                "summary": article.get("excerpt", "")[:280],
                "topic": "Regulatory and compliance update",
            }
            for article in articles[:10]
        ],
    })


def run_pipeline() -> Dict[str, Any]:
    html = fetch_homepage()
    articles = parse_articles(html)
    payload = create_raw_payload(articles)
    raw_uri = upload_to_gcs(payload, "raw")

    prompt = build_prompt(articles)
    llm_response = generate_llm_json(prompt, articles)
    result = normalize_llm_output(llm_response)

    output_payload = {"source": BASE_URL, "fetched_at": payload["fetched_at"], **result}
    output_uri = upload_to_gcs(output_payload, "output")

    if raw_uri or output_uri:
        result["cloud_storage"] = {
            "raw": raw_uri,
            "output": output_uri,
        }
    return result


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
