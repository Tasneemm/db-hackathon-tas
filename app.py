import json
import os
import logging
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
    import vertexai
    from vertexai.generative_models import GenerativeModel
except Exception:  # pragma: no cover - optional dependency
    vertexai = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0"}


BASE_URL = "https://artificialintelligenceact.eu/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_PREFIX = os.getenv("GCS_PREFIX", "ai-act-data")
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "europe-west1")
GCP_MODEL = os.getenv("GCP_MODEL", "gemini-2.5-flash")


def fetch_homepage() -> str:
    logger.info("Fetching homepage from %s", BASE_URL)
    response = requests.get(BASE_URL, timeout=20)
    response.raise_for_status()
    logger.info("Successfully fetched homepage.")
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
    except Exception as e:
        logger.error("Failed to fetch article text from %s: %s", url, e)
        return ""


def parse_articles(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    logger.info("Parsing articles from homepage HTML.")
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
    logger.info("Found %d unique articles to process.", len(unique[:20]))
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
        logger.warning("GCS_BUCKET not configured or google-cloud-storage not installed. Skipping upload.")
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
    # Prefer Google Cloud environment if project is set and library is available.
    # This works for both local ADC (gcloud auth application-default login)
    # and the service account environment on Cloud Run.
    if GCP_PROJECT and vertexai is not None:
        logger.info("Using Google Vertex AI (Gemini) for analysis.")
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        model = GenerativeModel(GCP_MODEL)
        response = model.generate_content(prompt)
        return response.text

    if os.getenv("OPENAI_API_KEY") and OpenAI is not None:
        logger.info("Using OpenAI API for analysis.")
        client = OpenAI() # API key is read from OPENAI_API_KEY env var by default
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    logger.warning("No LLM provider configured. Returning fallback JSON.")
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


def persist_processed_output(output_payload: Dict[str, Any]) -> None:
    logger.info("Loading processed output into BigQuery.i START")
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "llm_output.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    load_processed_output_to_bigquery(output_payload)


def load_processed_output_to_bigquery(output_payload: Dict[str, Any]) -> None:
    logger.info("Loading processed output into BigQuery.")
    try:
        import importlib
        load_to_bq = importlib.import_module("load_to_bq")
        logger.info("Imported load_to_bq module successfully.")
        load_to_bq.main()
    except Exception as exc:
        logger.warning("BigQuery loader could not be invoked: %s", exc)

def run_pipeline() -> Dict[str, Any]:
    html = fetch_homepage()
    articles = parse_articles(html)
    raw_payload = create_raw_payload(articles)
    raw_uri = upload_to_gcs(raw_payload, "raw")

    processed_results = []
    logger.info("Starting iterative processing of %d articles.", len(articles))
    i=1
    for article in articles:
        if i > 2 :
            break
        logger.info("-> Processing article: %s", article.get('title'))
        i += 1
        # Build a prompt for each individual article
        print("Print i  value",i)
        print("article value for prompt", article)
        prompt = build_prompt([article])
        print("Prompt value",prompt)
        llm_response = generate_llm_json(prompt, [article])
        print("LLM value",llm_response)
        try:
            # Normalize and store the structured output
            processed_article = normalize_llm_output(llm_response)
            processed_article['original_article'] = {
                'title': article.get('title'),
                'link': article.get('link')
            }
            processed_results.append(processed_article)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error("Failed to decode LLM output for article '%s': %s", article.get('link'), e)
            processed_results.append({
                'error': 'Failed to process article, LLM output was not valid JSON.',
                'original_article': {'title': article.get('title'), 'link': article.get('link')},
                'llm_response': llm_response
            })

    final_result = {"processed_articles": processed_results}
    print('Final result',final_result)
    output_payload = {"source": BASE_URL, "fetched_at": raw_payload["fetched_at"], **final_result}
    print('Print output payload',output_payload)
    output_uri = upload_to_gcs(output_payload, "output")
    print('uri',output_uri)
    #persist_processed_output(output_payload)
    print('Complete...........')
    logger.info("Running load_to_bq.py after pipeline completion.")
    try:
        import subprocess
        import sys
        subprocess.run([sys.executable, "load_to_bq.py"], check=True)
        logger.info("Completed load_to_bq.py execution.")
    except Exception as exc:
        logger.warning("Failed to run load_to_bq.py: %s", exc)
    if raw_uri or output_uri:
        final_result["cloud_storage"] = {
            "raw": raw_uri,
            "output": output_uri,
        }
    return final_result

if __name__ == "__main__":
    result = run_pipeline()
    #print(json.dumps(result, indent=2))
