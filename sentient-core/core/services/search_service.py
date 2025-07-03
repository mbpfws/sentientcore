"""Enhanced Search Service for Multi-Agent System

Provides unified search interface across multiple providers:
- Tavily: Real-time web search with AI-powered summarization
- Exa: Semantic search for technical content
- DuckDuckGo: Privacy-focused general web search
"""

import asyncio
import aiohttp
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import os
from urllib.parse import quote_plus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchProvider(Enum):
    """Available search providers"""
    TAVILY = "tavily"
    EXA = "exa"
    DUCKDUCKGO = "duckduckgo"

class SearchType(Enum):
    """Types of search queries"""
    GENERAL = "general"
    TECHNICAL = "technical"
    RESEARCH = "research"
    CODE = "code"
    DOCUMENTATION = "documentation"

@dataclass
class SearchResult:
    """Search result data structure"""
    title: str
    url: str
    content: str
    snippet: str
    score: float
    provider: SearchProvider
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class SearchQuery:
    """Search query configuration"""
    query: str
    search_type: SearchType = SearchType.GENERAL
    max_results: int = 10
    include_domains: List[str] = None
    exclude_domains: List[str] = None
    date_range: str = None  # "day", "week", "month", "year"
    language: str = "en"
    
    def __post_init__(self):
        if self.include_domains is None:
            self.include_domains = []
        if self.exclude_domains is None:
            self.exclude_domains = []

class BaseSearchProvider(ABC):
    """Abstract base class for search providers"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = None
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search with the provider"""
        pass
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = asyncio.get_event_loop().time()

class TavilyProvider(BaseSearchProvider):
    """Tavily search provider for real-time web search"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.tavily.com"
        self.rate_limit_delay = 0.5  # Tavily allows higher rate limits
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using Tavily API"""
        await self._rate_limit()
        
        # Prepare request payload
        payload = {
            "api_key": self.api_key,
            "query": query.query,
            "search_depth": "advanced" if query.search_type == SearchType.RESEARCH else "basic",
            "include_answer": True,
            "include_raw_content": True,
            "max_results": min(query.max_results, 20),  # Tavily max is 20
            "include_domains": query.include_domains,
            "exclude_domains": query.exclude_domains
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_tavily_results(data, query)
                else:
                    logger.error(f"Tavily API error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Tavily search error: {str(e)}")
            return []
    
    def _parse_tavily_results(self, data: Dict[str, Any], query: SearchQuery) -> List[SearchResult]:
        """Parse Tavily API response"""
        results = []
        
        for item in data.get("results", []):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("raw_content", ""),
                snippet=item.get("content", ""),
                score=item.get("score", 0.0),
                provider=SearchProvider.TAVILY,
                metadata={
                    "published_date": item.get("published_date"),
                    "search_type": query.search_type.value,
                    "answer": data.get("answer")  # AI-generated answer
                }
            )
            results.append(result)
        
        return results

class ExaProvider(BaseSearchProvider):
    """Exa search provider for semantic technical search"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.exa.ai"
        self.rate_limit_delay = 1.0
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using Exa API"""
        await self._rate_limit()
        
        # Prepare request payload
        payload = {
            "query": query.query,
            "type": "neural",  # Use neural search for better semantic understanding
            "useAutoprompt": True,
            "numResults": min(query.max_results, 10),  # Exa max is 10
            "contents": {
                "text": True,
                "highlights": True,
                "summary": True
            }
        }
        
        # Add domain filters if specified
        if query.include_domains:
            payload["includeDomains"] = query.include_domains
        if query.exclude_domains:
            payload["excludeDomains"] = query.exclude_domains
        
        # Add date filter if specified
        if query.date_range:
            payload["startPublishedDate"] = self._get_date_filter(query.date_range)
        
        try:
            async with self.session.post(
                f"{self.base_url}/search",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_exa_results(data, query)
                else:
                    logger.error(f"Exa API error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Exa search error: {str(e)}")
            return []
    
    def _parse_exa_results(self, data: Dict[str, Any], query: SearchQuery) -> List[SearchResult]:
        """Parse Exa API response"""
        results = []
        
        for item in data.get("results", []):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("text", ""),
                snippet=item.get("summary", item.get("text", "")[:300]),
                score=item.get("score", 0.0),
                provider=SearchProvider.EXA,
                metadata={
                    "published_date": item.get("publishedDate"),
                    "author": item.get("author"),
                    "highlights": item.get("highlights", []),
                    "search_type": query.search_type.value
                }
            )
            results.append(result)
        
        return results
    
    def _get_date_filter(self, date_range: str) -> str:
        """Convert date range to ISO format for Exa API"""
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        if date_range == "day":
            start_date = now - timedelta(days=1)
        elif date_range == "week":
            start_date = now - timedelta(weeks=1)
        elif date_range == "month":
            start_date = now - timedelta(days=30)
        elif date_range == "year":
            start_date = now - timedelta(days=365)
        else:
            start_date = now - timedelta(days=30)  # Default to month
        
        return start_date.isoformat()

class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search provider for privacy-focused search"""
    
    def __init__(self):
        super().__init__()  # No API key needed
        self.base_url = "https://api.duckduckgo.com"
        self.rate_limit_delay = 2.0  # Be respectful with rate limiting
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using DuckDuckGo Instant Answer API"""
        await self._rate_limit()
        
        # DuckDuckGo Instant Answer API
        params = {
            "q": query.query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        try:
            async with self.session.get(
                self.base_url,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_duckduckgo_results(data, query)
                else:
                    logger.error(f"DuckDuckGo API error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            return []
    
    def _parse_duckduckgo_results(self, data: Dict[str, Any], query: SearchQuery) -> List[SearchResult]:
        """Parse DuckDuckGo API response"""
        results = []
        
        # Abstract (instant answer)
        if data.get("Abstract"):
            result = SearchResult(
                title=data.get("Heading", "DuckDuckGo Instant Answer"),
                url=data.get("AbstractURL", ""),
                content=data.get("Abstract", ""),
                snippet=data.get("Abstract", "")[:300],
                score=1.0,
                provider=SearchProvider.DUCKDUCKGO,
                metadata={
                    "type": "instant_answer",
                    "source": data.get("AbstractSource"),
                    "search_type": query.search_type.value
                }
            )
            results.append(result)
        
        # Related topics
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                result = SearchResult(
                    title=topic.get("Text", "").split(" - ")[0],
                    url=topic.get("FirstURL", ""),
                    content=topic.get("Text", ""),
                    snippet=topic.get("Text", "")[:300],
                    score=0.8,
                    provider=SearchProvider.DUCKDUCKGO,
                    metadata={
                        "type": "related_topic",
                        "search_type": query.search_type.value
                    }
                )
                results.append(result)
        
        return results[:query.max_results]

class SearchService:
    """Unified search service with multiple providers
    
    Provides intelligent search routing and result aggregation across
    multiple search providers based on query type and content.
    """
    
    def __init__(self):
        """Initialize search service with available providers"""
        self.providers = {}
        self.provider_preferences = {
            SearchType.GENERAL: [SearchProvider.TAVILY, SearchProvider.DUCKDUCKGO],
            SearchType.TECHNICAL: [SearchProvider.EXA, SearchProvider.TAVILY],
            SearchType.RESEARCH: [SearchProvider.TAVILY, SearchProvider.EXA],
            SearchType.CODE: [SearchProvider.EXA, SearchProvider.TAVILY],
            SearchType.DOCUMENTATION: [SearchProvider.EXA, SearchProvider.TAVILY]
        }
        
        # Performance tracking
        self.performance_stats = {
            "total_searches": 0,
            "provider_usage": {provider.value: 0 for provider in SearchProvider},
            "avg_response_time": 0.0,
            "success_rate": 0.0
        }
        
        # Initialize providers based on available API keys
        self._initialize_providers()
        
        logger.info(f"SearchService initialized with providers: {list(self.providers.keys())}")
    
    def _initialize_providers(self):
        """Initialize search providers based on available API keys"""
        # Tavily provider
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            self.providers[SearchProvider.TAVILY] = TavilyProvider(tavily_key)
            logger.info("Tavily provider initialized")
        
        # Exa provider
        exa_key = os.getenv("EXA_API_KEY")
        if exa_key:
            self.providers[SearchProvider.EXA] = ExaProvider(exa_key)
            logger.info("Exa provider initialized")
        
        # DuckDuckGo provider (no API key needed)
        self.providers[SearchProvider.DUCKDUCKGO] = DuckDuckGoProvider()
        logger.info("DuckDuckGo provider initialized")
    
    async def search(self, query: Union[str, SearchQuery], 
                   providers: List[SearchProvider] = None,
                   aggregate_results: bool = True) -> List[SearchResult]:
        """Perform search across specified providers
        
        Args:
            query: Search query string or SearchQuery object
            providers: List of providers to use (default: auto-select based on query type)
            aggregate_results: Whether to combine and deduplicate results
            
        Returns:
            List of search results
        """
        # Convert string query to SearchQuery object
        if isinstance(query, str):
            query = SearchQuery(query=query)
        
        # Auto-select providers if not specified
        if providers is None:
            providers = self._select_providers(query.search_type)
        
        # Filter to only available providers
        available_providers = [p for p in providers if p in self.providers]
        
        if not available_providers:
            logger.warning("No available providers for search")
            return []
        
        # Perform searches concurrently
        search_tasks = []
        for provider in available_providers:
            provider_instance = self.providers[provider]
            task = self._search_with_provider(provider_instance, query)
            search_tasks.append(task)
        
        # Wait for all searches to complete
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine results
        all_results = []
        for i, results in enumerate(search_results):
            if isinstance(results, Exception):
                logger.error(f"Search failed for provider {available_providers[i]}: {results}")
                continue
            
            all_results.extend(results)
            self.performance_stats["provider_usage"][available_providers[i].value] += 1
        
        # Aggregate and deduplicate if requested
        if aggregate_results:
            all_results = self._aggregate_results(all_results)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit results
        final_results = all_results[:query.max_results]
        
        self.performance_stats["total_searches"] += 1
        
        logger.info(f"Search completed: {len(final_results)} results from {len(available_providers)} providers")
        return final_results
    
    async def _search_with_provider(self, provider: BaseSearchProvider, 
                                  query: SearchQuery) -> List[SearchResult]:
        """Perform search with a single provider"""
        async with provider:
            return await provider.search(query)
    
    def _select_providers(self, search_type: SearchType) -> List[SearchProvider]:
        """Select optimal providers for search type"""
        return self.provider_preferences.get(search_type, [SearchProvider.TAVILY])
    
    def _aggregate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Aggregate and deduplicate search results"""
        # Simple deduplication by URL
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    async def search_knowledge(self, query: str, search_type: SearchType = SearchType.RESEARCH,
                             max_results: int = 10) -> List[SearchResult]:
        """Specialized search for knowledge acquisition
        
        Optimized for research and technical content discovery.
        """
        search_query = SearchQuery(
            query=query,
            search_type=search_type,
            max_results=max_results,
            include_domains=[
                "github.com", "stackoverflow.com", "docs.python.org",
                "developer.mozilla.org", "arxiv.org", "medium.com"
            ]
        )
        
        return await self.search(search_query)
    
    async def search_code(self, query: str, language: str = None,
                        max_results: int = 10) -> List[SearchResult]:
        """Specialized search for code examples and documentation"""
        # Enhance query with language if specified
        enhanced_query = f"{query} {language}" if language else query
        
        search_query = SearchQuery(
            query=enhanced_query,
            search_type=SearchType.CODE,
            max_results=max_results,
            include_domains=[
                "github.com", "stackoverflow.com", "codepen.io",
                "replit.com", "codesandbox.io"
            ]
        )
        
        return await self.search(search_query)
    
    async def search_documentation(self, library: str, topic: str = None,
                                 max_results: int = 10) -> List[SearchResult]:
        """Specialized search for library and framework documentation"""
        query = f"{library} documentation"
        if topic:
            query += f" {topic}"
        
        search_query = SearchQuery(
            query=query,
            search_type=SearchType.DOCUMENTATION,
            max_results=max_results,
            include_domains=[
                "docs.python.org", "developer.mozilla.org", "reactjs.org",
                "vuejs.org", "angular.io", "fastapi.tiangolo.com",
                "flask.palletsprojects.com", "django.readthedocs.io"
            ]
        )
        
        return await self.search(search_query)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get search service performance statistics"""
        return self.performance_stats.copy()
    
    def get_available_providers(self) -> List[SearchProvider]:
        """Get list of available search providers"""
        return list(self.providers.keys())