import json
import unittest

from app import build_gcs_blob_name, build_prompt, normalize_llm_output


class PipelineTests(unittest.TestCase):
    def test_build_prompt_mentions_regulatory_compliance(self):
        posts = [{
            "title": "AI Act transparency rules",
            "date": "2026-07-01",
            "link": "https://example.com/article",
            "excerpt": "A new article about transparency and compliance."
        }]
        prompt = build_prompt(posts)
        self.assertIn("regulatory", prompt.lower())
        self.assertIn("compliance", prompt.lower())

    def test_normalize_llm_output_returns_expected_schema(self):
        raw_text = json.dumps({
            "source": "https://artificialintelligenceact.eu/",
            "items": [{
                "title": "Transparency rules",
                "date": "2026-07-01",
                "url": "https://example.com/article",
                "summary": "A summary",
                "topic": "Regulatory guidance"
            }]
        })
        result = normalize_llm_output(raw_text)
        self.assertEqual(result["source"], "https://artificialintelligenceact.eu/")
        self.assertIn("items", result)
        self.assertEqual(len(result["items"]), 1)

    def test_build_gcs_blob_name_uses_expected_path(self):
        name = build_gcs_blob_name("raw", "2026-07-16")
        self.assertEqual(name, "ai-act-data/raw/2026-07-16/raw_articles.json")


if __name__ == "__main__":
    unittest.main()
