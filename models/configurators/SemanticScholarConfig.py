# Semantic Scholar API Configuration
class SemanticScholarConfig:
    """Configuration class for Semantic Scholar API"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    API_KEY = "PkEo5fAuu37zVik3hcGuG8se7wlMgD1D2UfWTm2V"  # Set your API key here or use environment variable

    # Request headers
    @property
    def headers(self):
        headers = {
            "User-Agent": "KnowledgeFabric/1.0 (mailto:your-email@domain.com)",
            "Accept": "application/json"
        }
        if self.API_KEY:
            headers["x-api-key"] = self.API_KEY
        return headers

    # Rate limiting based on API key availability
    @property
    def request_delay(self):
        if self.API_KEY:
            return 0.31  # ~3 requests per second (1000/5min = 3.33/sec)
        else:
            return 3.1  # ~0.33 requests per second (100/5min = 0.33/sec)

    # Common field sets for different queries
    PAPER_FIELDS = [
        "paperId", "externalIds", "title", "abstract", "venue", "year",
        "referenceCount", "citationCount", "influentialCitationCount",
        "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "s2FieldsOfStudy",
        "authors", "citations", "references", "embedding", "tldr"
    ]

    AUTHOR_FIELDS = [
        "authorId", "externalIds", "name", "aliases", "affiliations",
        "homepage", "paperCount", "citationCount", "hIndex", "papers"
    ]