"""Web Search Tools for Groq Agentic Tooling

Provides web search capabilities that can be used with Groq's compound-beta models
for agentic research workflows.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import logging

from ..services.search_service import SearchService, SearchQuery, SearchType, SearchProvider
from ..services.enhanced_llm_service_main import AgenticTool

logger = logging.getLogger(__name__)

class WebSearchTools:
    """Web search tools for Groq's agentic models"""
    
    def __init__(self):
        self.search_service = SearchService()
    
    def get_tool_functions(self):
        """Return list of tool functions for registration with LLM service"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Perform a general web search for broad information gathering",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "technical_search",
                    "description": "Perform a technical search for detailed, specialized information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The technical search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 8)",
                                "default": 8
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "research_search",
                    "description": "Perform an academic/research-focused search for in-depth analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The research query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 12)",
                                "default": 12
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create tool definitions for Groq agentic tooling"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information using multiple search providers. Returns comprehensive results with content, snippets, and metadata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to execute"
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["general", "technical", "research", "code", "documentation"],
                                "description": "Type of search to perform",
                                "default": "research"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 20
                            },
                            "include_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Domains to include in search results"
                            },
                            "exclude_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Domains to exclude from search results"
                            },
                            "providers": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["tavily", "exa", "duckduckgo"]
                                },
                                "description": "Specific search providers to use"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "technical_search",
                    "description": "Specialized search for technical content, documentation, and code examples. Optimized for developer queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The technical search query"
                            },
                            "technology": {
                                "type": "string",
                                "description": "Specific technology or framework to focus on"
                            },
                            "content_type": {
                                "type": "string",
                                "enum": ["documentation", "code", "tutorials", "best_practices"],
                                "description": "Type of technical content to search for",
                                "default": "documentation"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 8,
                                "minimum": 1,
                                "maximum": 15
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "research_search",
                    "description": "Deep research search with AI-powered summarization. Best for comprehensive research tasks and analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The research query to investigate"
                            },
                            "research_depth": {
                                "type": "string",
                                "enum": ["basic", "advanced", "comprehensive"],
                                "description": "Depth of research to perform",
                                "default": "advanced"
                            },
                            "focus_areas": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific areas or aspects to focus the research on"
                            },
                            "time_range": {
                                "type": "string",
                                "enum": ["day", "week", "month", "year", "all"],
                                "description": "Time range for search results",
                                "default": "month"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 15,
                                "minimum": 5,
                                "maximum": 20
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def web_search(self, query: str, search_type: str = "research", 
                        max_results: int = 10, include_domains: List[str] = None,
                        exclude_domains: List[str] = None, providers: List[str] = None) -> Dict[str, Any]:
        """Execute web search with specified parameters"""
        try:
            # Convert string search_type to enum
            search_type_enum = SearchType(search_type)
            
            # Convert provider strings to enums if specified
            provider_enums = None
            if providers:
                provider_enums = [SearchProvider(p) for p in providers]
            
            # Create search query
            search_query = SearchQuery(
                query=query,
                search_type=search_type_enum,
                max_results=max_results,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or []
            )
            
            # Execute search
            results = await self.search_service.search(
                search_query, 
                providers=provider_enums
            )
            
            # Format results for tool response
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "content": result.content[:1000] if result.content else "",  # Limit content length
                    "snippet": result.snippet,
                    "score": result.score,
                    "provider": result.provider.value,
                    "metadata": result.metadata,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None
                })
            
            return {
                "success": True,
                "query": query,
                "search_type": search_type,
                "total_results": len(formatted_results),
                "results": formatted_results,
                "providers_used": list(set([r["provider"] for r in formatted_results]))
            }
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }
    
    async def technical_search(self, query: str, technology: str = None, 
                              content_type: str = "documentation", max_results: int = 8) -> Dict[str, Any]:
        """Execute technical search optimized for developer content"""
        try:
            # Enhance query with technology context
            enhanced_query = query
            if technology:
                enhanced_query = f"{technology} {query}"
            
            # Map content type to search type
            search_type_map = {
                "documentation": SearchType.DOCUMENTATION,
                "code": SearchType.CODE,
                "tutorials": SearchType.TECHNICAL,
                "best_practices": SearchType.TECHNICAL
            }
            
            search_type = search_type_map.get(content_type, SearchType.TECHNICAL)
            
            # Create search query with technical focus
            search_query = SearchQuery(
                query=enhanced_query,
                search_type=search_type,
                max_results=max_results,
                include_domains=[
                    "github.com", "stackoverflow.com", "docs.python.org",
                    "developer.mozilla.org", "reactjs.org", "nodejs.org"
                ] if content_type in ["documentation", "code"] else []
            )
            
            # Use EXA and Tavily for technical searches
            preferred_providers = [SearchProvider.EXA, SearchProvider.TAVILY]
            
            results = await self.search_service.search(
                search_query,
                providers=preferred_providers
            )
            
            # Format results with technical metadata
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "content": result.content[:800] if result.content else "",
                    "snippet": result.snippet,
                    "score": result.score,
                    "provider": result.provider.value,
                    "technology": technology,
                    "content_type": content_type,
                    "metadata": result.metadata
                })
            
            return {
                "success": True,
                "query": enhanced_query,
                "technology": technology,
                "content_type": content_type,
                "total_results": len(formatted_results),
                "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Technical search error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }
    
    async def research_search(self, query: str, research_depth: str = "advanced",
                             focus_areas: List[str] = None, time_range: str = "month",
                             max_results: int = 15) -> Dict[str, Any]:
        """Execute comprehensive research search with AI summarization"""
        try:
            # Enhance query with focus areas
            enhanced_query = query
            if focus_areas:
                focus_context = " ".join(focus_areas)
                enhanced_query = f"{query} {focus_context}"
            
            # Create research query
            search_query = SearchQuery(
                query=enhanced_query,
                search_type=SearchType.RESEARCH,
                max_results=max_results,
                date_range=time_range if time_range != "all" else None
            )
            
            # Use all available providers for comprehensive research
            results = await self.search_service.search(search_query)
            
            # Format results with research metadata
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.title,
                    "url": result.url,
                    "content": result.content[:1200] if result.content else "",
                    "snippet": result.snippet,
                    "score": result.score,
                    "provider": result.provider.value,
                    "research_depth": research_depth,
                    "focus_areas": focus_areas,
                    "metadata": result.metadata,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None
                })
            
            # Extract key insights from results
            insights = self._extract_research_insights(formatted_results)
            
            return {
                "success": True,
                "query": enhanced_query,
                "research_depth": research_depth,
                "focus_areas": focus_areas,
                "time_range": time_range,
                "total_results": len(formatted_results),
                "results": formatted_results,
                "insights": insights,
                "providers_used": list(set([r["provider"] for r in formatted_results]))
            }
            
        except Exception as e:
            logger.error(f"Research search error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }
    
    def _extract_research_insights(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key insights from research results"""
        insights = {
            "total_sources": len(results),
            "provider_distribution": {},
            "key_domains": [],
            "content_summary": ""
        }
        
        # Provider distribution
        for result in results:
            provider = result.get("provider", "unknown")
            insights["provider_distribution"][provider] = insights["provider_distribution"].get(provider, 0) + 1
        
        # Key domains
        domains = []
        for result in results:
            url = result.get("url", "")
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    domains.append(domain)
                except:
                    pass
        
        # Count domain frequency
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Get top 5 domains
        insights["key_domains"] = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return insights
    
    def get_tool_functions(self) -> List[AgenticTool]:
        """Get tool functions for registration with LLM service"""
        return [
            AgenticTool(
                name="web_search",
                description="Search the web for information using multiple search providers",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                },
                function=self.web_search,
                async_function=True
            ),
            AgenticTool(
                name="technical_search",
                description="Specialized search for technical content and documentation",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The technical search query"
                        },
                        "technology": {
                            "type": "string",
                            "description": "Specific technology or framework to focus on"
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["documentation", "code", "tutorials", "best_practices"],
                            "description": "Type of technical content to search for",
                            "default": "documentation"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 8)",
                            "default": 8
                        }
                    },
                    "required": ["query"]
                },
                function=self.technical_search,
                async_function=True
            ),
            AgenticTool(
                name="research_search",
                description="Comprehensive research search with AI summarization and insights",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The research query"
                        },
                        "research_depth": {
                            "type": "string",
                            "enum": ["basic", "advanced", "comprehensive"],
                            "description": "Depth of research to perform",
                            "default": "advanced"
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific areas or aspects to focus on"
                        },
                        "time_range": {
                            "type": "string",
                            "enum": ["day", "week", "month", "year", "all"],
                            "description": "Time range for search results",
                            "default": "month"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 15)",
                            "default": 15
                        }
                    },
                    "required": ["query"]
                },
                function=self.research_search,
                async_function=True
            )
        ]
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for Groq agentic tooling"""
        return self.tools