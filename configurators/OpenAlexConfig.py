# OpenAlex API Configuration
class OpenAlexConfig:
    """Configuration class for OpenAlex API"""

    BASE_URL = "https://api.openalex.org"

    # Common headers - add your email for higher rate limits
    HEADERS = {
        "User-Agent": "KnowledgeFabric/1.0",
        "Accept": "application/json"
    }

    # Rate limiting - be polite to the API
    REQUEST_DELAY = 1.1  # seconds between requests (slightly above 1/sec)

    # Common parameters
    DEFAULT_PARAMS = {
        "per-page": 25,  # results per page (max 200)
    }