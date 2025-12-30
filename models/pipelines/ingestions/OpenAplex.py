from models.configurators import OpenAlexConfig

class OpenAplex():

    def make_openalex_request(endpoint: str, params: Dict = None, delay: bool = True) -> Dict:
        """
        Make a request to the OpenAlex API with error handling and rate limiting.

        Args:
            endpoint: API endpoint (e.g., 'works', 'authors', 'institutions')
            params: Query parameters
            delay: Whether to add delay for rate limiting

        Returns:
            JSON response as dictionary
        """
        if params is None:
            params = {}

        # Merge with default parameters
        final_params = {**config.DEFAULT_PARAMS, **params}

        # Build URL
        url = urljoin(config.BASE_URL, endpoint)

        try:
            # Rate limiting
            if delay:
                time.sleep(config.REQUEST_DELAY)

            # Make request
            response = requests.get(url, headers=config.HEADERS, params=final_params)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            return None

    def test_api_connection():
        """Test the API connection with a simple request"""
        print("Testing OpenAlex API connection...")

        response = make_openalex_request("works", {"filter": "publication_year:2023", "per-page": 1})

        if response and "results" in response:
            print("✅ API connection successful!")
            print(f"Total works in 2023: {response['meta']['count']:,}")
            return True
        else:
            print("❌ API connection failed!")
            return False

