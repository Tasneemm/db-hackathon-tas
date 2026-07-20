import json
import os
import logging
from typing import List, Dict, Any

from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GCP_PROJECT = os.getenv("GCP_PROJECT")
BQ_DATASET = os.getenv("BQ_DATASET", "regulatory_analysis")
BQ_TABLE = os.getenv("BQ_TABLE", "ai_act_articles")
SOURCE_FILE = os.path.join(os.path.dirname(__file__), "data", "llm_output.json")

# Define the BigQuery table schema
SCHEMA = [
    bigquery.SchemaField("regulation_name", "STRING"),
    bigquery.SchemaField("article_number", "STRING"),
    bigquery.SchemaField("effective_date", "STRING"),
    bigquery.SchemaField("compliance_deadline", "STRING"),
    bigquery.SchemaField("summary", "STRING"),
    bigquery.SchemaField("mandatory_or_advisory", "STRING"),
    bigquery.SchemaField("regulatory_authority", "STRING"),
    bigquery.SchemaField("applicable_geography", "STRING"),
    bigquery.SchemaField("overall_risk_score", "FLOAT"),
    bigquery.SchemaField("risk_category", "STRING"),
    bigquery.SchemaField("impacted_banking_domains", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("domain_name", "STRING"),
        bigquery.SchemaField("confidence_score", "INTEGER"),
        bigquery.SchemaField("reason", "STRING"),
    ]),
    bigquery.SchemaField("risk_dimensions", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("dimension", "STRING"),
        bigquery.SchemaField("score", "INTEGER"),
        bigquery.SchemaField("reason", "STRING"),
    ]),
    bigquery.SchemaField("original_article_title", "STRING"),
    bigquery.SchemaField("original_article_link", "STRING"),
    bigquery.SchemaField("analysis_status", "STRING"),
    bigquery.SchemaField("error_message", "STRING"),
]


def get_value(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Gets a value from a nested dict using a list of possible keys."""
    for key in keys:
        if isinstance(data, dict) and key in data and data[key] is not None:
            return data[key]
    return default


def normalize_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes a single article analysis into a consistent schema."""
    # Handle various nested structures for regulation details
    reg_details = get_value(article, ['Regulation_Analysis', 'RegulationIdentification', 'Analysis', 'regulatory_analysis', 'Regulation_Details']) or article

    # Handle different keys for risk dimensions
    risk_dims_raw = get_value(article, ['RiskDimensions', 'RiskDimensionEvaluation', 'Risk_Dimensions_Evaluation', 'risk_dimensions']) or {}
    risk_dimensions = []
    if isinstance(risk_dims_raw, dict):
        for dim, values in risk_dims_raw.items():
            if isinstance(values, dict):
                risk_dimensions.append({
                    "dimension": dim,
                    "score": get_value(values, ['Score']),
                    "reason": get_value(values, ['Reason', 'Reasoning'])
                })

    # Handle different keys for impacted domains
    impacted_domains_raw = get_value(article, ['ImpactedBankingDomains', 'Impacted_Banking_Domains', 'Banking_Domains_Impacted', 'impacted_banking_domains']) or []
    impacted_banking_domains = []
    if isinstance(impacted_domains_raw, list):
        for domain in impacted_domains_raw:
            impacted_banking_domains.append({
                "domain_name": get_value(domain, ['DomainName', 'Domain_Name', 'domain_name']),
                "confidence_score": get_value(domain, ['ConfidenceScore', 'Confidence_Score', 'confidence_score']),
                "reason": get_value(domain, ['Reason', 'reason'])
            })

    # Handle different keys for overall risk
    overall_risk = get_value(article, ['OverallRiskAssessment', 'Overall_Risk_Assessment', 'OverallRisk']) or {}

    return {
        "regulation_name": get_value(reg_details, ['RegulationName', 'Regulation_Name', 'regulation_name']),
        "article_number": get_value(reg_details, ['ArticleNumber', 'Article_Number', 'article_number']),
        "effective_date": get_value(reg_details, ['EffectiveDate', 'Effective_Date', 'effective_date']),
        "compliance_deadline": get_value(reg_details, ['ComplianceDeadline', 'compliance_deadline']),
        "summary": get_value(reg_details, ['Summary', 'summary']),
        "mandatory_or_advisory": get_value(reg_details, ['MandatoryOrAdvisory', 'Mandatory_or_Advisory', 'mandatory_or_advisory', 'RequirementMandatoryOrAdvisory']),
        "regulatory_authority": get_value(reg_details, ['RegulatoryAuthority', 'Regulatory_Authority', 'regulatory_authority']),
        "applicable_geography": get_value(reg_details, ['ApplicableGeography', 'applicable_geography']),
        "overall_risk_score": get_value(overall_risk, ['OverallRiskScore', 'overall_risk_score']) or get_value(article, ['OverallRiskScore', 'overall_risk_score']),
        "risk_category": get_value(overall_risk, ['RiskCategory', 'risk_category']) or get_value(article, ['RiskCategory', 'risk_category']),
        "impacted_banking_domains": impacted_banking_domains,
        "risk_dimensions": risk_dimensions,
        "original_article_title": get_value(article.get('original_article', {}), ['title']),
        "original_article_link": get_value(article.get('original_article', {}), ['link']),
        "analysis_status": get_value(article, ['AnalysisStatus']),
        "error_message": get_value(article, ['ErrorMessage']),
    }


def normalize_data(raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reads the raw JSON data and normalizes all processed articles."""
    normalized_articles = []
    for article in raw_data.get("processed_articles", []):
        normalized_articles.append(normalize_article(article))
    return normalized_articles


def create_bigquery_dataset_and_table(client: bigquery.Client, dataset_id: str, table_id: str, schema: List[bigquery.SchemaField]):
    """Ensures the BigQuery dataset and table exist."""
    full_dataset_id = f"{client.project}.{dataset_id}"
    dataset = bigquery.Dataset(full_dataset_id)
    try:
        client.get_dataset(dataset_id)
        logger.info("Dataset %s already exists.", dataset_id)
    except Exception:
        logger.info("Creating dataset %s.", dataset_id)
        client.create_dataset(dataset, timeout=30)

    full_table_id = f"{full_dataset_id}.{table_id}"
    table = bigquery.Table(full_table_id, schema=schema)
    try:
        client.get_table(full_table_id)
        logger.info("Table %s already exists.", full_table_id)
    except Exception:
        logger.info("Creating table %s.", full_table_id)
        client.create_table(table)


def load_data_to_bq(client: bigquery.Client, data: List[Dict[str, Any]], table_id: str):
    """Loads the normalized data into the specified BigQuery table."""
    if not data:
        logger.warning("No data to load. Skipping BigQuery load.")
        return

    job_config = bigquery.LoadJobConfig(
        schema=SCHEMA,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Overwrite table with new data
    )

    try:
        load_job = client.load_table_from_json(data, table_id, job_config=job_config)
        logger.info("Starting BigQuery load job %s", load_job.job_id)

        load_job.result()  # Wait for the job to complete

        destination_table = client.get_table(table_id)
        logger.info("Loaded %d rows to table %s.", destination_table.num_rows, table_id)
    except Exception as e:
        logger.error("Failed to load data to BigQuery: %s", e)
        if hasattr(e, 'errors'):
            logger.error("BigQuery errors: %s", e.errors)
        raise


def main():
    """Main function to run the data loading pipeline."""
    if not GCP_PROJECT:
        logger.error("GCP_PROJECT environment variable not set. Exiting.")
        return

    logger.info("Starting BigQuery loading process for AI Act analysis.")

    # 1. Read and normalize the data
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        logger.error("Source file not found: %s", SOURCE_FILE)
        return
    except json.JSONDecodeError:
        logger.error("Could not decode JSON from source file: %s", SOURCE_FILE)
        return

    normalized_data = normalize_data(raw_data)
    logger.info("Successfully normalized %d articles.", len(normalized_data))

    # 2. Set up BigQuery client and table
    client = bigquery.Client(project=GCP_PROJECT)
    table_ref = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    create_bigquery_dataset_and_table(client, BQ_DATASET, BQ_TABLE, SCHEMA)

    # 3. Load data into BigQuery
    load_data_to_bq(client, normalized_data, table_ref)

    logger.info("BigQuery loading process finished successfully.")


if __name__ == "__main__":
    main()
