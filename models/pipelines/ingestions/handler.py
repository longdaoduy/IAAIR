from models.configurators import OpenAlexConfig
from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Author import Author
from models.schemas.edges.CitedBy import CitedBy

class OpenAplex():
    config: OpenAlexConfig
    nums_papers_to_pull: int
    def __init__(self):
        self.nums_papers_to_pull = 1000

    def pull_OpenAlex_Paper(self):
        # TO DO:
        # pull 1000 papers from OpenAlex API postman request 'https://api.openalex.org/works'
        return