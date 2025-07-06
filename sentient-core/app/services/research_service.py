from typing import Dict, List, Optional, Any, AsyncGenerator, Union
import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import re
from pathlib import Path
import aiohttp
import hashlib

# Search provider imports
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

try:
    from exa_py import Exa
except ImportError:
    Exa = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

from .memory_service import MemoryService, MemoryType
from .llm_service import EnhancedLLMService, ChatMessage, MessageRole
from .sse_manager import SSEConnectionManager, EventType


class SearchProvider(Enum):
    """Supported search providers"""
    TAVILY = "tavily"
    EXA = "exa"
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    BING = "bing"


class ResearchStatus(Enum):
    """Research task status"""
    PENDING = "pending"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContentType(Enum):
    """Types of content to search for"""
    GENERAL = "general"
    ACADEMIC = "academic"
    NEWS = "news"
    TECHNICAL = "technical"
    SOCIAL = "social"
    IMAGES = "images"
    VIDEOS = "videos"


@dataclass
class SearchResult:
    """Individual search result"""
    title: str
    url: str
    content: str
    snippet: str
    provider: SearchProvider
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchQuery:
    """Research query configuration"""
    query: str
    providers: List[SearchProvider] = field(default_factory=lambda: [SearchProvider.TAVILY])
    content_types: List[ContentType] = field(default_factory=lambda: [ContentType.GENERAL])
    max_results: int = 10
    language: str = "en"
    region: Optional[str] = None
    time_range: Optional[str] = None  # "day", "week", "month", "year"
    include_domains: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchTask:
    """Research task definition"""
    id: str
    query: ResearchQuery
    session_id: Optional[str] = None
    status: ResearchStatus = ResearchStatus.PENDING
    results: List[SearchResult] = field(default_factory=list)
    analysis: Optional[str] = None
    synthesis: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchConfig:
    """Configuration for research service"""
    tavily_api_key: Optional[str] = None
    exa_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    bing_api_key: Optional[str] = None
    max_concurrent_searches: int = 5
    default_timeout: int = 30
    cache_results: bool = True
    cache_duration_hours: int = 24
    enable_content_extraction: bool = True
    max_content_length: int = 10000


class EnhancedResearchService:
    """Advanced research service with multiple providers and intelligent analysis"""
    
    def __init__(
        self,
        config: ResearchConfig,
        memory_service: MemoryService,
        llm_service: EnhancedLLMService,
        sse_manager: SSEConnectionManager
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Core services
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.sse_manager = sse_manager
        
        # Search providers
        self._providers: Dict[SearchProvider, Any] = {}
        self._initialize_providers()
        
        # Task management
        self._active_tasks: Dict[str, ResearchTask] = {}
        self._task_history: List[ResearchTask] = []
        
        # Caching
        self._result_cache: Dict[str, List[SearchResult]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Performance tracking
        self._stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "cache_hits": 0,
            "average_search_time": 0.0,
            "total_results_found": 0
        }
        
        # Concurrency control
        self._search_semaphore = asyncio.Semaphore(config.max_concurrent_searches)
    
    def _initialize_providers(self):
        """Initialize search providers"""
        try:
            if self.config.tavily_api_key and TavilyClient:
                self._providers[SearchProvider.TAVILY] = TavilyClient(api_key=self.config.tavily_api_key)
                self.logger.info("Initialized Tavily search provider")
        except Exception as e:
            self.logger.error(f"Failed to initialize Tavily: {e}")
        
        try:
            if self.config.exa_api_key and Exa:
                self._providers[SearchProvider.EXA] = Exa(api_key=self.config.exa_api_key)
                self.logger.info("Initialized Exa search provider")
        except Exception as e:
            self.logger.error(f"Failed to initialize Exa: {e}")
        
        try:
            if DDGS:
                self._providers[SearchProvider.DUCKDUCKGO] = DDGS()
                self.logger.info("Initialized DuckDuckGo search provider")
        except Exception as e:
            self.logger.error(f"Failed to initialize DuckDuckGo: {e}")
    
    async def start_research(
        self,
        query: ResearchQuery,
        session_id: Optional[str] = None
    ) -> str:
        """Start a new research task"""
        task_id = str(uuid.uuid4())
        
        task = ResearchTask(
            id=task_id,
            query=query,
            session_id=session_id,
            metadata={
                "query_text": query.query,
                "providers": [p.value for p in query.providers],
                "max_results": query.max_results
            }
        )
        
        self._active_tasks[task_id] = task
        
        # Start research in background
        asyncio.create_task(self._execute_research(task_id))
        
        # Notify start
        await self.sse_manager.broadcast_research_progress(
            task_id,
            {
                "status": task.status.value,
                "progress": 0.0,
                "query": query.query
            },
            session_id
        )
        
        self.logger.info(f"Started research task: {task_id}")
        return task_id
    
    async def _execute_research(self, task_id: str):
        """Execute research task"""
        task = self._active_tasks[task_id]
        
        try:
            task.started_at = datetime.now()
            
            # Phase 1: Search
            task.status = ResearchStatus.SEARCHING
            task.progress = 0.1
            await self._notify_progress(task)
            
            search_results = await self._perform_searches(task)
            task.results = search_results
            task.progress = 0.4
            await self._notify_progress(task)
            
            # Phase 2: Analysis
            task.status = ResearchStatus.ANALYZING
            task.progress = 0.5
            await self._notify_progress(task)
            
            analysis = await self._analyze_results(task)
            task.analysis = analysis
            task.progress = 0.7
            await self._notify_progress(task)
            
            # Phase 3: Synthesis
            task.status = ResearchStatus.SYNTHESIZING
            task.progress = 0.8
            await self._notify_progress(task)
            
            synthesis = await self._synthesize_findings(task)
            task.synthesis = synthesis
            task.progress = 1.0
            
            # Complete
            task.status = ResearchStatus.COMPLETED
            task.completed_at = datetime.now()
            await self._notify_progress(task)
            
            # Store results
            await self._store_research_results(task)
            
            # Update stats
            self._update_stats(True, len(search_results))
            
        except Exception as e:
            task.status = ResearchStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            self.logger.error(f"Research task failed: {e}")
            
            await self.sse_manager.broadcast_error(
                f"Research failed: {e}",
                task.session_id
            )
            
            self._update_stats(False, 0)
        
        finally:
            # Move to history
            self._task_history.append(task)
            self._active_tasks.pop(task_id, None)
            
            # Limit history size
            if len(self._task_history) > 1000:
                self._task_history = self._task_history[-500:]
    
    async def _perform_searches(self, task: ResearchTask) -> List[SearchResult]:
        """Perform searches across multiple providers"""
        query = task.query
        
        # Check cache first
        cache_key = self._generate_cache_key(query)
        if self.config.cache_results and cache_key in self._result_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).total_seconds() < (self.config.cache_duration_hours * 3600):
                self.logger.debug(f"Cache hit for query: {query.query}")
                self._stats["cache_hits"] += 1
                return self._result_cache[cache_key]
        
        # Perform searches
        search_tasks = []
        for provider in query.providers:
            if provider in self._providers:
                search_tasks.append(self._search_with_provider(provider, query))
        
        if not search_tasks:
            raise ValueError("No available search providers")
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine and filter results
        all_results = []
        for result in search_results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Search provider error: {result}")
        
        # Remove duplicates and rank
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(unique_results, query.query)
        
        # Limit results
        final_results = ranked_results[:query.max_results]
        
        # Cache results
        if self.config.cache_results:
            self._result_cache[cache_key] = final_results
            self._cache_timestamps[cache_key] = datetime.now()
        
        return final_results
    
    async def _search_with_provider(
        self,
        provider: SearchProvider,
        query: ResearchQuery
    ) -> List[SearchResult]:
        """Search with specific provider"""
        async with self._search_semaphore:
            try:
                if provider == SearchProvider.TAVILY:
                    return await self._search_tavily(query)
                elif provider == SearchProvider.EXA:
                    return await self._search_exa(query)
                elif provider == SearchProvider.DUCKDUCKGO:
                    return await self._search_duckduckgo(query)
                else:
                    self.logger.warning(f"Unsupported provider: {provider}")
                    return []
            
            except Exception as e:
                self.logger.error(f"Search failed with {provider.value}: {e}")
                return []
    
    async def _search_tavily(self, query: ResearchQuery) -> List[SearchResult]:
        """Search using Tavily"""
        client = self._providers[SearchProvider.TAVILY]
        
        search_params = {
            "query": query.query,
            "search_depth": "advanced",
            "max_results": min(query.max_results, 20),
            "include_answer": True,
            "include_raw_content": self.config.enable_content_extraction
        }
        
        if query.include_domains:
            search_params["include_domains"] = query.include_domains
        if query.exclude_domains:
            search_params["exclude_domains"] = query.exclude_domains
        
        response = await asyncio.to_thread(client.search, **search_params)
        
        results = []
        for item in response.get("results", []):
            content = item.get("raw_content", "") or item.get("content", "")
            if len(content) > self.config.max_content_length:
                content = content[:self.config.max_content_length] + "..."
            
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=content,
                snippet=item.get("content", "")[:500],
                provider=SearchProvider.TAVILY,
                relevance_score=item.get("score", 0.0),
                metadata={
                    "published_date": item.get("published_date"),
                    "domain": item.get("domain")
                }
            )
            results.append(result)
        
        return results
    
    async def _search_exa(self, query: ResearchQuery) -> List[SearchResult]:
        """Search using Exa"""
        client = self._providers[SearchProvider.EXA]
        
        search_params = {
            "query": query.query,
            "num_results": min(query.max_results, 20),
            "use_autoprompt": True,
            "type": "neural"
        }
        
        if query.include_domains:
            search_params["include_domains"] = query.include_domains
        if query.exclude_domains:
            search_params["exclude_domains"] = query.exclude_domains
        
        response = await asyncio.to_thread(client.search, **search_params)
        
        results = []
        for item in response.results:
            # Get content if enabled
            content = ""
            if self.config.enable_content_extraction:
                try:
                    content_response = await asyncio.to_thread(
                        client.get_contents,
                        [item.id]
                    )
                    if content_response.results:
                        content = content_response.results[0].text or ""
                        if len(content) > self.config.max_content_length:
                            content = content[:self.config.max_content_length] + "..."
                except Exception as e:
                    self.logger.debug(f"Failed to get content for {item.url}: {e}")
            
            result = SearchResult(
                title=item.title or "",
                url=item.url or "",
                content=content,
                snippet=(item.snippet or "")[:500],
                provider=SearchProvider.EXA,
                relevance_score=item.score or 0.0,
                metadata={
                    "published_date": getattr(item, 'published_date', None),
                    "author": getattr(item, 'author', None)
                }
            )
            results.append(result)
        
        return results
    
    async def _search_duckduckgo(self, query: ResearchQuery) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        ddgs = self._providers[SearchProvider.DUCKDUCKGO]
        
        search_params = {
            "keywords": query.query,
            "max_results": min(query.max_results, 20),
            "region": query.region or "wt-wt",
            "safesearch": "moderate"
        }
        
        if query.time_range:
            search_params["timelimit"] = query.time_range
        
        # Perform search
        search_results = await asyncio.to_thread(
            lambda: list(ddgs.text(**search_params))
        )
        
        results = []
        for item in search_results:
            # Extract content if enabled
            content = ""
            if self.config.enable_content_extraction:
                try:
                    content = await self._extract_content(item.get("href", ""))
                except Exception as e:
                    self.logger.debug(f"Failed to extract content from {item.get('href')}: {e}")
            
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                content=content,
                snippet=item.get("body", "")[:500],
                provider=SearchProvider.DUCKDUCKGO,
                relevance_score=0.5,  # DDG doesn't provide scores
                metadata={}
            )
            results.append(result)
        
        return results
    
    async def _extract_content(self, url: str) -> str:
        """Extract content from URL"""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Simple content extraction (could be enhanced with BeautifulSoup)
                        content = re.sub(r'<[^>]+>', '', html)
                        content = re.sub(r'\s+', ' ', content).strip()
                        
                        if len(content) > self.config.max_content_length:
                            content = content[:self.config.max_content_length] + "..."
                        
                        return content
        except Exception as e:
            self.logger.debug(f"Content extraction failed for {url}: {e}")
        
        return ""
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank results by relevance"""
        query_terms = set(query.lower().split())
        
        for result in results:
            # Calculate relevance score based on multiple factors
            title_score = self._calculate_text_relevance(result.title.lower(), query_terms)
            snippet_score = self._calculate_text_relevance(result.snippet.lower(), query_terms)
            
            # Combine scores
            combined_score = (
                title_score * 0.4 +
                snippet_score * 0.3 +
                result.relevance_score * 0.3
            )
            
            result.relevance_score = combined_score
        
        # Sort by relevance score (descending)
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _calculate_text_relevance(self, text: str, query_terms: set) -> float:
        """Calculate text relevance to query terms"""
        if not text or not query_terms:
            return 0.0
        
        text_terms = set(text.split())
        matches = len(query_terms.intersection(text_terms))
        
        return matches / len(query_terms) if query_terms else 0.0
    
    async def _analyze_results(self, task: ResearchTask) -> str:
        """Analyze search results using LLM"""
        if not task.results:
            return "No results found to analyze."
        
        # Prepare results summary for analysis
        results_summary = []
        for i, result in enumerate(task.results[:10], 1):  # Limit to top 10
            results_summary.append(f"""
            Result {i}:
            Title: {result.title}
            URL: {result.url}
            Snippet: {result.snippet}
            Relevance: {result.relevance_score:.2f}
            """)
        
        analysis_prompt = f"""
        Analyze the following search results for the query: "{task.query.query}"
        
        Search Results:
        {''.join(results_summary)}
        
        Please provide:
        1. Key themes and patterns identified
        2. Quality assessment of the sources
        3. Gaps or missing information
        4. Reliability and credibility notes
        5. Summary of main findings
        
        Focus on being analytical and objective.
        """
        
        messages = [ChatMessage(MessageRole.USER, analysis_prompt)]
        response = await self.llm_service.chat_completion(messages=messages)
        
        return response.content
    
    async def _synthesize_findings(self, task: ResearchTask) -> str:
        """Synthesize research findings"""
        if not task.results or not task.analysis:
            return "Insufficient data for synthesis."
        
        # Prepare content for synthesis
        content_pieces = []
        for result in task.results[:5]:  # Top 5 results
            if result.content:
                content_pieces.append(f"""
                Source: {result.title} ({result.url})
                Content: {result.content[:1000]}...
                """)
        
        synthesis_prompt = f"""
        Based on the research query "{task.query.query}" and the following analysis and content, 
        provide a comprehensive synthesis:
        
        Analysis:
        {task.analysis}
        
        Key Content:
        {''.join(content_pieces)}
        
        Please provide:
        1. Executive summary
        2. Detailed findings with supporting evidence
        3. Conclusions and implications
        4. Recommendations for further research
        5. Source citations
        
        Make it comprehensive yet accessible.
        """
        
        messages = [ChatMessage(MessageRole.USER, synthesis_prompt)]
        response = await self.llm_service.chat_completion(messages=messages)
        
        return response.content
    
    async def _store_research_results(self, task: ResearchTask):
        """Store research results in memory"""
        # Store main research task
        await self.memory_service.store_memory(
            content=json.dumps({
                "task_id": task.id,
                "query": task.query.query,
                "status": task.status.value,
                "results_count": len(task.results),
                "analysis": task.analysis[:500] if task.analysis else None,
                "synthesis": task.synthesis[:500] if task.synthesis else None,
                "execution_time": (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else None
            }),
            memory_type=MemoryType.RESEARCH,
            metadata={
                "task_id": task.id,
                "query": task.query.query,
                "type": "research_task",
                "status": task.status.value
            }
        )
        
        # Store individual results
        for i, result in enumerate(task.results[:10]):  # Store top 10
            await self.memory_service.store_memory(
                content=json.dumps({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "provider": result.provider.value,
                    "relevance_score": result.relevance_score
                }),
                memory_type=MemoryType.RESEARCH,
                metadata={
                    "task_id": task.id,
                    "result_index": i,
                    "type": "search_result",
                    "provider": result.provider.value
                }
            )
    
    async def _notify_progress(self, task: ResearchTask):
        """Notify progress via SSE"""
        await self.sse_manager.broadcast_research_progress(
            task.id,
            {
                "status": task.status.value,
                "progress": task.progress,
                "results_count": len(task.results),
                "current_phase": task.status.value.replace('_', ' ').title()
            },
            task.session_id
        )
    
    def _generate_cache_key(self, query: ResearchQuery) -> str:
        """Generate cache key for query"""
        key_data = {
            "query": query.query,
            "providers": sorted([p.value for p in query.providers]),
            "max_results": query.max_results,
            "language": query.language,
            "include_domains": sorted(query.include_domains),
            "exclude_domains": sorted(query.exclude_domains)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_stats(self, success: bool, results_count: int):
        """Update service statistics"""
        self._stats["total_searches"] += 1
        
        if success:
            self._stats["successful_searches"] += 1
            self._stats["total_results_found"] += results_count
        else:
            self._stats["failed_searches"] += 1
    
    # Public API methods
    async def get_research_status(self, task_id: str) -> Optional[ResearchTask]:
        """Get research task status"""
        return self._active_tasks.get(task_id)
    
    async def cancel_research(self, task_id: str) -> bool:
        """Cancel research task"""
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.status = ResearchStatus.CANCELLED
            task.completed_at = datetime.now()
            return True
        return False
    
    async def list_active_research(self) -> List[Dict[str, Any]]:
        """List active research tasks"""
        return [
            {
                "id": task.id,
                "query": task.query.query,
                "status": task.status.value,
                "progress": task.progress,
                "results_count": len(task.results),
                "started_at": task.started_at.isoformat() if task.started_at else None
            }
            for task in self._active_tasks.values()
        ]
    
    async def get_research_results(
        self,
        task_id: str,
        include_content: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get research results"""
        # Check active tasks first
        task = self._active_tasks.get(task_id)
        if not task:
            # Check history
            task = next((t for t in self._task_history if t.id == task_id), None)
        
        if not task:
            return None
        
        results_data = []
        for result in task.results:
            result_data = {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "provider": result.provider.value,
                "relevance_score": result.relevance_score,
                "timestamp": result.timestamp.isoformat()
            }
            
            if include_content:
                result_data["content"] = result.content
            
            results_data.append(result_data)
        
        return {
            "id": task.id,
            "query": task.query.query,
            "status": task.status.value,
            "progress": task.progress,
            "results": results_data,
            "analysis": task.analysis,
            "synthesis": task.synthesis,
            "error": task.error,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "metadata": task.metadata
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self._stats,
            "available_providers": [p.value for p in self._providers.keys()],
            "active_tasks": len(self._active_tasks),
            "task_history": len(self._task_history),
            "cache_size": len(self._result_cache),
            "cache_enabled": self.config.cache_results
        }
    
    async def clear_cache(self):
        """Clear result cache"""
        self._result_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Research cache cleared")
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old task history"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        original_count = len(self._task_history)
        self._task_history = [
            task for task in self._task_history
            if task.completed_at and task.completed_at > cutoff_time
        ]
        
        cleaned_count = original_count - len(self._task_history)
        self.logger.info(f"Cleaned up {cleaned_count} old research tasks")
        return cleaned_count