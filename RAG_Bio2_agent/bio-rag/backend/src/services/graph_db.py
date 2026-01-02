"""Graph Database Service - Neo4j Integration for GraphRAG"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PaperNode:
    """Paper node in the graph"""
    pmid: str
    title: str
    abstract: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None


@dataclass
class AuthorNode:
    """Author node in the graph"""
    name: str
    affiliation: Optional[str] = None


@dataclass
class KeywordNode:
    """Keyword/MeSH term node in the graph"""
    term: str
    mesh_id: Optional[str] = None


@dataclass
class CitationRelation:
    """Citation relationship between papers"""
    citing_pmid: str
    cited_pmid: str
    context: Optional[str] = None  # Citation context text


@dataclass
class GraphSearchResult:
    """Result from graph-based search"""
    pmid: str
    title: str
    relevance_score: float
    path_type: str  # 'direct_citation', 'co_citation', 'keyword_related', etc.
    path_length: int
    related_entities: List[Dict[str, Any]]


class GraphDBService:
    """
    Neo4j Graph Database Service for Bio-RAG

    Manages paper citations, author collaborations, and keyword relationships
    for enhanced RAG retrieval using graph traversal.
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        self.uri = uri or getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or getattr(settings, 'NEO4J_USER', 'neo4j')
        self.password = password or getattr(settings, 'NEO4J_PASSWORD', 'password')
        self._driver = None
        self._async_driver = None

    def _get_driver(self):
        """Get or create sync Neo4j driver"""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                # Verify connectivity
                self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.uri}")
            except (ServiceUnavailable, AuthError) as e:
                logger.warning(f"Neo4j connection failed: {e}. Using mock mode.")
                self._driver = None
        return self._driver

    async def _get_async_driver(self):
        """Get or create async Neo4j driver"""
        if self._async_driver is None:
            try:
                self._async_driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                logger.info(f"Async connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.warning(f"Async Neo4j connection failed: {e}")
                self._async_driver = None
        return self._async_driver

    def close(self):
        """Close the driver connection"""
        if self._driver:
            self._driver.close()
            self._driver = None

    async def close_async(self):
        """Close async driver connection"""
        if self._async_driver:
            await self._async_driver.close()
            self._async_driver = None

    # ==================== Schema Setup ====================

    def setup_schema(self):
        """Create indexes and constraints for optimal performance"""
        driver = self._get_driver()
        if not driver:
            logger.warning("Neo4j not available, skipping schema setup")
            return

        with driver.session() as session:
            # Create constraints (unique identifiers)
            constraints = [
                "CREATE CONSTRAINT paper_pmid IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
                "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
                "CREATE CONSTRAINT keyword_term IF NOT EXISTS FOR (k:Keyword) REQUIRE k.term IS UNIQUE",
            ]

            # Create indexes for faster lookups
            indexes = [
                "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
                "CREATE INDEX paper_date IF NOT EXISTS FOR (p:Paper) ON (p.publication_date)",
                "CREATE INDEX keyword_mesh IF NOT EXISTS FOR (k:Keyword) ON (k.mesh_id)",
            ]

            for query in constraints + indexes:
                try:
                    session.run(query)
                except Exception as e:
                    logger.debug(f"Schema query skipped (may already exist): {e}")

            logger.info("Neo4j schema setup complete")

    # ==================== Paper Operations ====================

    def create_paper(self, paper: PaperNode) -> bool:
        """Create or update a paper node"""
        driver = self._get_driver()
        if not driver:
            return False

        query = """
        MERGE (p:Paper {pmid: $pmid})
        SET p.title = $title,
            p.abstract = $abstract,
            p.journal = $journal,
            p.publication_date = $publication_date,
            p.updated_at = datetime()
        RETURN p
        """

        with driver.session() as session:
            result = session.run(
                query,
                pmid=paper.pmid,
                title=paper.title,
                abstract=paper.abstract,
                journal=paper.journal,
                publication_date=paper.publication_date
            )
            return result.single() is not None

    def create_papers_batch(self, papers: List[PaperNode]) -> int:
        """Batch create/update paper nodes"""
        driver = self._get_driver()
        if not driver:
            return 0

        query = """
        UNWIND $papers AS paper
        MERGE (p:Paper {pmid: paper.pmid})
        SET p.title = paper.title,
            p.abstract = paper.abstract,
            p.journal = paper.journal,
            p.publication_date = paper.publication_date,
            p.updated_at = datetime()
        RETURN count(p) as count
        """

        papers_data = [
            {
                'pmid': p.pmid,
                'title': p.title,
                'abstract': p.abstract,
                'journal': p.journal,
                'publication_date': p.publication_date
            }
            for p in papers
        ]

        with driver.session() as session:
            result = session.run(query, papers=papers_data)
            record = result.single()
            return record['count'] if record else 0

    # ==================== Citation Operations ====================

    def create_citation(self, citation: CitationRelation) -> bool:
        """Create a citation relationship between papers"""
        driver = self._get_driver()
        if not driver:
            return False

        query = """
        MATCH (citing:Paper {pmid: $citing_pmid})
        MATCH (cited:Paper {pmid: $cited_pmid})
        MERGE (citing)-[r:CITES]->(cited)
        SET r.context = $context,
            r.created_at = datetime()
        RETURN r
        """

        with driver.session() as session:
            result = session.run(
                query,
                citing_pmid=citation.citing_pmid,
                cited_pmid=citation.cited_pmid,
                context=citation.context
            )
            return result.single() is not None

    def create_citations_batch(self, citations: List[CitationRelation]) -> int:
        """Batch create citation relationships"""
        driver = self._get_driver()
        if not driver:
            return 0

        # First ensure all papers exist
        query = """
        UNWIND $citations AS cit
        MERGE (citing:Paper {pmid: cit.citing_pmid})
        MERGE (cited:Paper {pmid: cit.cited_pmid})
        MERGE (citing)-[r:CITES]->(cited)
        SET r.context = cit.context,
            r.created_at = datetime()
        RETURN count(r) as count
        """

        citations_data = [
            {
                'citing_pmid': c.citing_pmid,
                'cited_pmid': c.cited_pmid,
                'context': c.context
            }
            for c in citations
        ]

        with driver.session() as session:
            result = session.run(query, citations=citations_data)
            record = result.single()
            return record['count'] if record else 0

    # ==================== Author Operations ====================

    def link_author_to_paper(self, author_name: str, pmid: str, position: int = 0) -> bool:
        """Link an author to a paper"""
        driver = self._get_driver()
        if not driver:
            return False

        query = """
        MERGE (a:Author {name: $author_name})
        MERGE (p:Paper {pmid: $pmid})
        MERGE (a)-[r:AUTHORED]->(p)
        SET r.position = $position
        RETURN r
        """

        with driver.session() as session:
            result = session.run(
                query,
                author_name=author_name,
                pmid=pmid,
                position=position
            )
            return result.single() is not None

    # ==================== Keyword Operations ====================

    def link_keyword_to_paper(self, keyword: str, pmid: str, mesh_id: str = None) -> bool:
        """Link a keyword/MeSH term to a paper"""
        driver = self._get_driver()
        if not driver:
            return False

        query = """
        MERGE (k:Keyword {term: $keyword})
        SET k.mesh_id = COALESCE($mesh_id, k.mesh_id)
        MERGE (p:Paper {pmid: $pmid})
        MERGE (p)-[r:HAS_KEYWORD]->(k)
        RETURN r
        """

        with driver.session() as session:
            result = session.run(
                query,
                keyword=keyword,
                pmid=pmid,
                mesh_id=mesh_id
            )
            return result.single() is not None

    # ==================== Graph Search Operations ====================

    def find_citing_papers(self, pmid: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find papers that cite the given paper"""
        driver = self._get_driver()
        if not driver:
            return []

        query = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper {pmid: $pmid})
        RETURN citing.pmid AS pmid, citing.title AS title, citing.journal AS journal
        LIMIT $limit
        """

        with driver.session() as session:
            result = session.run(query, pmid=pmid, limit=limit)
            return [dict(record) for record in result]

    def find_cited_papers(self, pmid: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find papers cited by the given paper"""
        driver = self._get_driver()
        if not driver:
            return []

        query = """
        MATCH (citing:Paper {pmid: $pmid})-[:CITES]->(cited:Paper)
        RETURN cited.pmid AS pmid, cited.title AS title, cited.journal AS journal
        LIMIT $limit
        """

        with driver.session() as session:
            result = session.run(query, pmid=pmid, limit=limit)
            return [dict(record) for record in result]

    def find_co_cited_papers(self, pmid: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find papers that are frequently co-cited with the given paper"""
        driver = self._get_driver()
        if not driver:
            return []

        query = """
        MATCH (p1:Paper {pmid: $pmid})<-[:CITES]-(citing:Paper)-[:CITES]->(p2:Paper)
        WHERE p1 <> p2
        WITH p2, count(citing) AS co_citation_count
        ORDER BY co_citation_count DESC
        RETURN p2.pmid AS pmid, p2.title AS title, co_citation_count
        LIMIT $limit
        """

        with driver.session() as session:
            result = session.run(query, pmid=pmid, limit=limit)
            return [dict(record) for record in result]

    def find_related_by_keywords(self, pmid: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find papers sharing keywords with the given paper"""
        driver = self._get_driver()
        if not driver:
            return []

        query = """
        MATCH (p1:Paper {pmid: $pmid})-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(p2:Paper)
        WHERE p1 <> p2
        WITH p2, collect(k.term) AS shared_keywords, count(k) AS keyword_count
        ORDER BY keyword_count DESC
        RETURN p2.pmid AS pmid, p2.title AS title, shared_keywords, keyword_count
        LIMIT $limit
        """

        with driver.session() as session:
            result = session.run(query, pmid=pmid, limit=limit)
            return [dict(record) for record in result]

    def find_author_network(self, author_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Find co-authors and their papers"""
        driver = self._get_driver()
        if not driver:
            return []

        query = """
        MATCH (a1:Author {name: $author_name})-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
        WHERE a1 <> a2
        WITH a2, collect(p.title) AS collaborations, count(p) AS collab_count
        ORDER BY collab_count DESC
        RETURN a2.name AS co_author, collaborations, collab_count
        LIMIT $limit
        """

        with driver.session() as session:
            result = session.run(query, author_name=author_name, limit=limit)
            return [dict(record) for record in result]

    def find_citation_path(self, from_pmid: str, to_pmid: str, max_depth: int = 4) -> List[Dict[str, Any]]:
        """Find citation path between two papers"""
        driver = self._get_driver()
        if not driver:
            return []

        query = """
        MATCH path = shortestPath(
            (p1:Paper {pmid: $from_pmid})-[:CITES*1..$max_depth]-(p2:Paper {pmid: $to_pmid})
        )
        RETURN [node IN nodes(path) | {pmid: node.pmid, title: node.title}] AS path,
               length(path) AS path_length
        """

        with driver.session() as session:
            result = session.run(
                query,
                from_pmid=from_pmid,
                to_pmid=to_pmid,
                max_depth=max_depth
            )
            record = result.single()
            return record['path'] if record else []

    # ==================== GraphRAG Search ====================

    def graph_enhanced_search(
        self,
        seed_pmids: List[str],
        max_depth: int = 2,
        limit: int = 20
    ) -> List[GraphSearchResult]:
        """
        Perform graph-enhanced search starting from seed papers

        This combines:
        1. Direct citations (papers citing or cited by seeds)
        2. Co-citations (papers frequently cited together)
        3. Keyword relationships (papers sharing keywords)
        4. Author connections (papers by same/related authors)
        """
        driver = self._get_driver()
        if not driver:
            return []

        query = """
        UNWIND $seed_pmids AS seed_pmid
        MATCH (seed:Paper {pmid: seed_pmid})

        // Find related papers through various paths
        CALL {
            WITH seed
            // Direct citations
            MATCH (seed)-[:CITES*1..2]-(related:Paper)
            WHERE seed <> related
            RETURN related, 'citation' AS path_type, 1.0 AS base_score

            UNION

            WITH seed
            // Keyword relationships
            MATCH (seed)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(related:Paper)
            WHERE seed <> related
            RETURN related, 'keyword' AS path_type, 0.8 AS base_score

            UNION

            WITH seed
            // Same author
            MATCH (seed)<-[:AUTHORED]-(a:Author)-[:AUTHORED]->(related:Paper)
            WHERE seed <> related
            RETURN related, 'author' AS path_type, 0.7 AS base_score
        }

        WITH related, path_type, base_score, count(*) AS connection_count
        WITH related, path_type, base_score * (1 + log(connection_count)) AS score
        ORDER BY score DESC

        RETURN DISTINCT related.pmid AS pmid,
               related.title AS title,
               score AS relevance_score,
               path_type,
               collect(path_type) AS path_types
        LIMIT $limit
        """

        with driver.session() as session:
            result = session.run(
                query,
                seed_pmids=seed_pmids,
                limit=limit
            )

            results = []
            for record in result:
                results.append(GraphSearchResult(
                    pmid=record['pmid'],
                    title=record['title'] or '',
                    relevance_score=record['relevance_score'],
                    path_type=record['path_type'],
                    path_length=1,
                    related_entities=[]
                ))

            return results

    # ==================== Visualization ====================

    # Korean to English medical term mapping
    KOREAN_MEDICAL_TERMS = {
        # Cancers
        '췌장암': 'pancreatic cancer',
        '폐암': 'lung cancer',
        '유방암': 'breast cancer',
        '대장암': 'colorectal cancer',
        '간암': 'liver cancer',
        '위암': 'gastric cancer',
        '전립선암': 'prostate cancer',
        '난소암': 'ovarian cancer',
        '백혈병': 'leukemia',
        '림프종': 'lymphoma',
        '암': 'cancer',
        '종양': 'tumor',
        # Treatments
        '면역요법': 'immunotherapy',
        '항암제': 'chemotherapy',
        '방사선치료': 'radiation therapy',
        '유전자치료': 'gene therapy',
        '표적치료': 'targeted therapy',
        '세포치료': 'cell therapy',
        # Technologies
        '크리스퍼': 'CRISPR',
        '유전자편집': 'gene editing',
        '유전자가위': 'CRISPR',
        '유전자': 'gene',
        'mRNA백신': 'mRNA vaccine',
        '단백질': 'protein',
        '항체': 'antibody',
        '바이오마커': 'biomarker',
        '딥러닝': 'deep learning',
        '인공지능': 'artificial intelligence',
        '머신러닝': 'machine learning',
        '신경망': 'neural network',
        # Diseases
        '당뇨병': 'diabetes',
        '알츠하이머': 'Alzheimer',
        '파킨슨': 'Parkinson',
        '심혈관': 'cardiovascular',
        '자가면역': 'autoimmune',
        '염증': 'inflammation',
        # General
        '줄기세포': 'stem cell',
        '임상시험': 'clinical trial',
        '바이러스': 'virus',
        '박테리아': 'bacteria',
        '감염': 'infection',
        '코로나': 'COVID',
        '백신': 'vaccine',
    }

    def _translate_korean_query(self, query: str) -> str:
        """Translate Korean medical terms to English for search"""
        if not query:
            return query

        # Check if query contains Korean characters
        import re
        if not re.search('[가-힣]', query):
            return query  # No Korean, return as-is

        # Try to find matching Korean term
        query_lower = query.lower().strip()
        for korean, english in self.KOREAN_MEDICAL_TERMS.items():
            if korean in query:
                return english

        # If no exact match, return original (will likely return no results)
        return query

    def get_visualization_data(
        self,
        query: str = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get graph data for visualization (nodes and edges)

        Returns nodes (papers, authors, keywords) and edges (citations, authorships, keyword links)
        in a format suitable for frontend 3D visualization.
        """
        driver = self._get_driver()
        if not driver:
            return {'nodes': [], 'edges': [], 'status': 'disconnected'}

        # Translate Korean query to English if needed
        if query:
            query = self._translate_korean_query(query)
            logger.info(f"Search query (after translation): {query}")

        # If query provided, search for related papers first
        if query:
            visualization_query = """
            // Find papers matching the query
            MATCH (p:Paper)
            WHERE p.title =~ $pattern OR p.abstract =~ $pattern
            WITH p LIMIT 10

            // Get related nodes
            OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
            OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
            OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
            OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)

            WITH collect(DISTINCT p) + collect(DISTINCT cited) + collect(DISTINCT citing) AS papers,
                 collect(DISTINCT a) AS authors,
                 collect(DISTINCT k) AS keywords

            UNWIND papers AS paper
            WITH DISTINCT paper, authors, keywords
            WHERE paper IS NOT NULL

            // Collect all nodes
            WITH collect({
                id: 'paper_' + paper.pmid,
                type: 'paper',
                label: CASE WHEN size(paper.title) > 40 THEN substring(paper.title, 0, 40) + '...' ELSE paper.title END,
                pmid: paper.pmid,
                title: paper.title,
                journal: paper.journal
            }) AS paperNodes, authors, keywords

            // Add author nodes
            UNWIND authors AS author
            WITH paperNodes, collect(DISTINCT {
                id: 'author_' + author.name,
                type: 'author',
                label: author.name,
                name: author.name
            }) AS authorNodes, keywords

            // Add keyword nodes
            UNWIND keywords AS keyword
            WITH paperNodes, authorNodes, collect(DISTINCT {
                id: 'keyword_' + keyword.term,
                type: 'keyword',
                label: keyword.term,
                term: keyword.term
            }) AS keywordNodes

            RETURN paperNodes + authorNodes + keywordNodes AS nodes
            """
            pattern = f"(?i).*{query}.*"
        else:
            # Get full graph overview (limited)
            visualization_query = """
            // Get papers
            MATCH (p:Paper)
            WITH p LIMIT $limit

            WITH collect({
                id: 'paper_' + p.pmid,
                type: 'paper',
                label: CASE WHEN size(p.title) > 40 THEN substring(p.title, 0, 40) + '...' ELSE p.title END,
                pmid: p.pmid,
                title: p.title,
                journal: p.journal
            }) AS paperNodes

            // Get authors (limited)
            MATCH (a:Author)
            WITH paperNodes, a LIMIT 30
            WITH paperNodes, collect({
                id: 'author_' + a.name,
                type: 'author',
                label: a.name,
                name: a.name
            }) AS authorNodes

            // Get keywords (limited)
            MATCH (k:Keyword)
            WITH paperNodes, authorNodes, k LIMIT 30
            WITH paperNodes, authorNodes, collect({
                id: 'keyword_' + k.term,
                type: 'keyword',
                label: k.term,
                term: k.term
            }) AS keywordNodes

            RETURN paperNodes + authorNodes + keywordNodes AS nodes
            """
            pattern = None

        # Get edges separately
        edges_query = """
        // Citation edges
        MATCH (p1:Paper)-[c:CITES]->(p2:Paper)
        WITH collect({
            source: 'paper_' + p1.pmid,
            target: 'paper_' + p2.pmid,
            type: 'cites',
            label: 'CITES'
        }) AS citationEdges

        // Authorship edges
        MATCH (a:Author)-[au:AUTHORED]->(p:Paper)
        WITH citationEdges, collect({
            source: 'author_' + a.name,
            target: 'paper_' + p.pmid,
            type: 'authored',
            label: 'AUTHORED'
        }) AS authorshipEdges

        // Keyword edges
        MATCH (p:Paper)-[h:HAS_KEYWORD]->(k:Keyword)
        WITH citationEdges, authorshipEdges, collect({
            source: 'paper_' + p.pmid,
            target: 'keyword_' + k.term,
            type: 'has_keyword',
            label: 'HAS_KEYWORD'
        }) AS keywordEdges

        RETURN citationEdges + authorshipEdges + keywordEdges AS edges
        """

        with driver.session() as session:
            try:
                # Get nodes
                if query:
                    node_result = session.run(visualization_query, pattern=pattern)
                else:
                    node_result = session.run(visualization_query, limit=limit)

                node_record = node_result.single()
                nodes = node_record['nodes'] if node_record else []

                # Get edges
                edge_result = session.run(edges_query)
                edge_record = edge_result.single()
                edges = edge_record['edges'] if edge_record else []

                # Filter edges to only include nodes we have
                node_ids = {n['id'] for n in nodes}
                filtered_edges = [
                    e for e in edges
                    if e['source'] in node_ids and e['target'] in node_ids
                ]

                return {
                    'nodes': nodes,
                    'edges': filtered_edges,
                    'status': 'connected',
                    'node_count': len(nodes),
                    'edge_count': len(filtered_edges)
                }

            except Exception as e:
                logger.error(f"Visualization query error: {e}")
                return {'nodes': [], 'edges': [], 'status': 'error', 'error': str(e)}

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get graph database statistics"""
        driver = self._get_driver()
        if not driver:
            return {'status': 'disconnected'}

        query = """
        MATCH (p:Paper) WITH count(p) AS papers
        MATCH (a:Author) WITH papers, count(a) AS authors
        MATCH (k:Keyword) WITH papers, authors, count(k) AS keywords
        MATCH ()-[c:CITES]->() WITH papers, authors, keywords, count(c) AS citations
        MATCH ()-[au:AUTHORED]->() WITH papers, authors, keywords, citations, count(au) AS authorships
        MATCH ()-[kw:HAS_KEYWORD]->()
        RETURN papers, authors, keywords, citations, authorships, count(kw) AS keyword_links
        """

        with driver.session() as session:
            result = session.run(query)
            record = result.single()

            if record:
                return {
                    'status': 'connected',
                    'papers': record['papers'],
                    'authors': record['authors'],
                    'keywords': record['keywords'],
                    'citations': record['citations'],
                    'authorships': record['authorships'],
                    'keyword_links': record['keyword_links']
                }

            return {'status': 'empty'}


# Global service instance
_graph_service: Optional[GraphDBService] = None


def get_graph_service() -> GraphDBService:
    """Get or create graph database service instance"""
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphDBService()
    return _graph_service
