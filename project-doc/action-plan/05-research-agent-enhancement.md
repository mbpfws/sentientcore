# 05 - Research Agent Enhancement

## Overview

The Research Agent serves as the knowledge acquisition and synthesis hub for the multi-agent system. It integrates multiple search providers (Tavily, Exa, Google), performs intelligent information gathering, synthesizes findings, and maintains a comprehensive knowledge base. This phase transforms the existing basic research agent into a sophisticated research and analysis system.

## Current State Analysis

### Existing File
- `core/agents/research_agent.py` - Basic research functionality

### Enhancement Requirements
- Multi-provider search integration (Tavily, Exa, Google)
- Intelligent query planning and execution
- Information synthesis and summarization
- Knowledge graph construction
- Real-time research monitoring
- Integration with memory service for knowledge persistence

## Implementation Tasks

### Task 5.1: Enhanced Research Agent

**File**: `core/agents/research_agent.py` (Complete Rewrite)

**Research Agent Implementation**:
```python
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
from enum import Enum
import json

from .base_agent import BaseAgent, AgentStatus
from ..services.search_service import SearchService
from ..services.memory_service import MemoryService
from ..services.llm_service import LLMService
from ..models import ResearchTask, SearchResult, KnowledgeNode

class ResearchType(Enum):
    TECHNICAL_RESEARCH = "technical_research"
    MARKET_RESEARCH = "market_research"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    DOCUMENTATION_SEARCH = "documentation_search"
    CODE_EXAMPLES = "code_examples"
    BEST_PRACTICES = "best_practices"
    TROUBLESHOOTING = "troubleshooting"
    TREND_ANALYSIS = "trend_analysis"

class ResearchAgent(BaseAgent):
    def __init__(self, agent_id: str = "research_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Research Agent",
            description="Advanced research and knowledge synthesis agent"
        )
        self.capabilities = [
            "web_search",
            "information_synthesis",
            "knowledge_extraction",
            "trend_analysis",
            "documentation_research",
            "competitive_analysis",
            "technical_research"
        ]
        
        self.search_service = None
        self.active_research_sessions = {}
        self.research_templates = {}
        self.knowledge_graph = {}
        
    async def initialize(self, search_service: SearchService):
        """Initialize research agent with search service"""
        self.search_service = search_service
        await self._load_research_templates()
        await self.update_status(AgentStatus.IDLE)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process research task and return comprehensive results"""
        try:
            await self.update_status(AgentStatus.THINKING)
            
            # Parse research task
            research_task = self._parse_research_task(task)
            
            # Create research session
            session_id = f"research_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.active_research_sessions[session_id] = {
                'task': research_task,
                'start_time': datetime.utcnow(),
                'status': 'active',
                'results': []
            }
            
            await self.update_status(AgentStatus.WORKING)
            
            # Execute research plan
            research_plan = await self._create_research_plan(research_task)
            results = await self._execute_research_plan(session_id, research_plan)
            
            # Synthesize findings
            synthesis = await self._synthesize_research_results(results)
            
            # Store knowledge
            await self._store_research_knowledge(session_id, synthesis)
            
            await self.update_status(AgentStatus.COMPLETED)
            
            return {
                'session_id': session_id,
                'research_type': research_task.research_type,
                'query': research_task.query,
                'synthesis': synthesis,
                'raw_results': results,
                'knowledge_nodes': synthesis.get('knowledge_nodes', []),
                'recommendations': synthesis.get('recommendations', []),
                'confidence_score': synthesis.get('confidence_score', 0.0)
            }
            
        except Exception as e:
            await self.update_status(AgentStatus.ERROR, str(e))
            raise
    
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Determine if agent can handle research task"""
        return task.get('type') in [
            'research_request',
            'information_gathering',
            'knowledge_synthesis',
            'documentation_search',
            'competitive_analysis',
            'technical_research'
        ]
    
    def _parse_research_task(self, task: Dict[str, Any]) -> 'ResearchTask':
        """Parse incoming task into structured research task"""
        return ResearchTask(
            query=task.get('query', ''),
            research_type=ResearchType(task.get('research_type', 'technical_research')),
            scope=task.get('scope', 'comprehensive'),
            depth=task.get('depth', 'medium'),
            time_range=task.get('time_range', 'recent'),
            sources=task.get('sources', ['web', 'documentation']),
            filters=task.get('filters', {}),
            context=task.get('context', {})
        )
    
    async def _create_research_plan(self, research_task: 'ResearchTask') -> Dict[str, Any]:
        """Create comprehensive research plan using LLM"""
        prompt = f"""
        Create a detailed research plan for the following research task:
        
        Query: {research_task.query}
        Research Type: {research_task.research_type.value}
        Scope: {research_task.scope}
        Depth: {research_task.depth}
        Time Range: {research_task.time_range}
        Available Sources: {research_task.sources}
        Context: {research_task.context}
        
        Create a research plan with the following structure:
        1. Primary search queries (3-5 specific queries)
        2. Secondary search queries (2-3 follow-up queries)
        3. Search strategies for each provider (Tavily, Exa, Google)
        4. Information extraction focus areas
        5. Synthesis approach
        6. Quality validation criteria
        
        Return as JSON with this structure:
        {{
            "primary_queries": ["query1", "query2", ...],
            "secondary_queries": ["query1", "query2", ...],
            "search_strategies": {{
                "tavily": {{"approach": "...", "parameters": {{}}}},
                "exa": {{"approach": "...", "parameters": {{}}}},
                "google": {{"approach": "...", "parameters": {{}}}}
            }},
            "extraction_focus": ["focus1", "focus2", ...],
            "synthesis_approach": "...",
            "validation_criteria": ["criteria1", "criteria2", ...]
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.3
        )
        
        try:
            plan = json.loads(response)
            return plan
        except json.JSONDecodeError:
            # Fallback to default plan
            return self._get_default_research_plan(research_task)
    
    async def _execute_research_plan(self, session_id: str, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute research plan across multiple search providers"""
        all_results = []
        
        # Execute primary queries
        for query in plan.get('primary_queries', []):
            await self.log_activity(f"Executing primary query: {query}")
            
            # Search with multiple providers
            search_tasks = []
            
            # Tavily search
            if 'tavily' in plan.get('search_strategies', {}):
                search_tasks.append(
                    self.search_service.search_tavily(
                        query,
                        **plan['search_strategies']['tavily'].get('parameters', {})
                    )
                )
            
            # Exa search
            if 'exa' in plan.get('search_strategies', {}):
                search_tasks.append(
                    self.search_service.search_exa(
                        query,
                        **plan['search_strategies']['exa'].get('parameters', {})
                    )
                )
            
            # Google search
            if 'google' in plan.get('search_strategies', {}):
                search_tasks.append(
                    self.search_service.search_google(
                        query,
                        **plan['search_strategies']['google'].get('parameters', {})
                    )
                )
            
            # Execute searches concurrently
            if search_tasks:
                results = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                for result in results:
                    if not isinstance(result, Exception) and result:
                        all_results.extend(result)
        
        # Execute secondary queries based on initial findings
        if plan.get('secondary_queries'):
            await self._execute_secondary_queries(session_id, plan, all_results)
        
        # Filter and rank results
        filtered_results = await self._filter_and_rank_results(all_results, plan)
        
        return filtered_results
    
    async def _execute_secondary_queries(self, session_id: str, plan: Dict[str, Any], initial_results: List[Dict[str, Any]]):
        """Execute secondary queries based on initial findings"""
        # Analyze initial results to refine secondary queries
        refined_queries = await self._refine_secondary_queries(
            plan.get('secondary_queries', []),
            initial_results
        )
        
        for query in refined_queries:
            await self.log_activity(f"Executing secondary query: {query}")
            
            # Execute with best-performing provider from initial results
            best_provider = self._determine_best_provider(initial_results)
            
            if best_provider == 'tavily':
                results = await self.search_service.search_tavily(query)
            elif best_provider == 'exa':
                results = await self.search_service.search_exa(query)
            else:
                results = await self.search_service.search_google(query)
            
            if results:
                initial_results.extend(results)
    
    async def _filter_and_rank_results(self, results: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter and rank search results based on relevance and quality"""
        # Remove duplicates
        unique_results = self._deduplicate_results(results)
        
        # Score results based on relevance, recency, and authority
        scored_results = []
        for result in unique_results:
            score = await self._calculate_result_score(result, plan)
            scored_results.append({
                **result,
                'relevance_score': score
            })
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Return top 20 results
        return scored_results[:20]
    
    async def _synthesize_research_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize research results into comprehensive knowledge"""
        # Extract key information from results
        extracted_info = await self._extract_key_information(results)
        
        # Create knowledge synthesis using LLM
        synthesis_prompt = f"""
        Synthesize the following research results into a comprehensive knowledge summary:
        
        Research Results:
        {json.dumps(extracted_info, indent=2)}
        
        Create a synthesis with the following structure:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (bullet points)
        3. Technical Details (if applicable)
        4. Best Practices and Recommendations
        5. Potential Challenges and Solutions
        6. Related Technologies/Concepts
        7. Confidence Assessment
        
        Return as JSON with this structure:
        {{
            "executive_summary": "...",
            "key_findings": ["finding1", "finding2", ...],
            "technical_details": {{
                "concepts": ["concept1", "concept2", ...],
                "implementations": ["impl1", "impl2", ...],
                "tools": ["tool1", "tool2", ...]
            }},
            "recommendations": ["rec1", "rec2", ...],
            "challenges": [
                {{"challenge": "...", "solution": "..."}}
            ],
            "related_concepts": ["concept1", "concept2", ...],
            "confidence_score": 0.85,
            "knowledge_nodes": [
                {{"concept": "...", "description": "...", "relationships": [...]}}
            ]
        }}
        """
        
        response = await self.llm_service.generate_response(
            synthesis_prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        try:
            synthesis = json.loads(response)
            
            # Add metadata
            synthesis['synthesis_timestamp'] = datetime.utcnow().isoformat()
            synthesis['source_count'] = len(results)
            synthesis['research_depth'] = self._calculate_research_depth(results)
            
            return synthesis
        except json.JSONDecodeError:
            # Fallback synthesis
            return self._create_fallback_synthesis(results)
    
    async def _extract_key_information(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key information from search results"""
        extracted = {
            'titles': [],
            'summaries': [],
            'key_points': [],
            'technical_terms': [],
            'urls': [],
            'publication_dates': []
        }
        
        for result in results:
            if result.get('title'):
                extracted['titles'].append(result['title'])
            
            if result.get('content') or result.get('snippet'):
                content = result.get('content', result.get('snippet', ''))
                extracted['summaries'].append(content[:500])  # Limit length
            
            if result.get('url'):
                extracted['urls'].append(result['url'])
            
            if result.get('published_date'):
                extracted['publication_dates'].append(result['published_date'])
        
        return extracted
    
    async def _store_research_knowledge(self, session_id: str, synthesis: Dict[str, Any]):
        """Store research knowledge in memory service"""
        # Store synthesis
        await self.memory_service.store_knowledge(
            'research_synthesis',
            synthesis,
            {
                'session_id': session_id,
                'research_type': synthesis.get('research_type'),
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        # Store knowledge nodes in knowledge graph
        for node in synthesis.get('knowledge_nodes', []):
            await self.memory_service.store_knowledge(
                'knowledge_graph',
                node,
                {
                    'session_id': session_id,
                    'node_type': 'research_concept',
                    'concept': node.get('concept')
                }
            )
    
    # Helper methods
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL and content similarity"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
            elif not url:  # Handle results without URLs
                unique_results.append(result)
        
        return unique_results
    
    async def _calculate_result_score(self, result: Dict[str, Any], plan: Dict[str, Any]) -> float:
        """Calculate relevance score for search result"""
        score = 0.0
        
        # Base relevance (0.0 - 0.4)
        if result.get('title'):
            score += 0.2
        if result.get('content') or result.get('snippet'):
            score += 0.2
        
        # Authority score (0.0 - 0.3)
        url = result.get('url', '')
        if any(domain in url for domain in ['github.com', 'stackoverflow.com', 'docs.', 'official']):
            score += 0.3
        elif any(domain in url for domain in ['.edu', '.org', 'medium.com']):
            score += 0.2
        else:
            score += 0.1
        
        # Recency score (0.0 - 0.3)
        if result.get('published_date'):
            try:
                pub_date = datetime.fromisoformat(result['published_date'])
                days_old = (datetime.utcnow() - pub_date).days
                if days_old <= 30:
                    score += 0.3
                elif days_old <= 180:
                    score += 0.2
                elif days_old <= 365:
                    score += 0.1
            except:
                score += 0.1
        
        return min(score, 1.0)
    
    def _determine_best_provider(self, results: List[Dict[str, Any]]) -> str:
        """Determine best performing search provider"""
        provider_scores = {'tavily': 0, 'exa': 0, 'google': 0}
        
        for result in results:
            provider = result.get('provider', 'unknown')
            score = result.get('relevance_score', 0)
            if provider in provider_scores:
                provider_scores[provider] += score
        
        return max(provider_scores, key=provider_scores.get)
    
    def _calculate_research_depth(self, results: List[Dict[str, Any]]) -> str:
        """Calculate research depth based on result count and quality"""
        if len(results) >= 15:
            return 'comprehensive'
        elif len(results) >= 8:
            return 'thorough'
        elif len(results) >= 4:
            return 'moderate'
        else:
            return 'basic'
    
    async def _refine_secondary_queries(self, queries: List[str], initial_results: List[Dict[str, Any]]) -> List[str]:
        """Refine secondary queries based on initial findings"""
        # Extract key terms from initial results
        key_terms = set()
        for result in initial_results[:5]:  # Use top 5 results
            title = result.get('title', '')
            content = result.get('content', result.get('snippet', ''))
            
            # Simple keyword extraction (could be enhanced with NLP)
            words = (title + ' ' + content).lower().split()
            technical_words = [w for w in words if len(w) > 6 and w.isalpha()]
            key_terms.update(technical_words[:3])
        
        # Enhance queries with key terms
        refined_queries = []
        for query in queries:
            if key_terms:
                enhanced_query = f"{query} {' '.join(list(key_terms)[:2])}"
                refined_queries.append(enhanced_query)
            else:
                refined_queries.append(query)
        
        return refined_queries
    
    def _get_default_research_plan(self, research_task: 'ResearchTask') -> Dict[str, Any]:
        """Get default research plan if LLM planning fails"""
        return {
            "primary_queries": [
                research_task.query,
                f"{research_task.query} best practices",
                f"{research_task.query} implementation guide"
            ],
            "secondary_queries": [
                f"{research_task.query} examples",
                f"{research_task.query} troubleshooting"
            ],
            "search_strategies": {
                "tavily": {"approach": "comprehensive", "parameters": {"max_results": 10}},
                "exa": {"approach": "semantic", "parameters": {"num_results": 8}},
                "google": {"approach": "broad", "parameters": {"num": 10}}
            },
            "extraction_focus": ["technical_details", "best_practices", "examples"],
            "synthesis_approach": "comprehensive_analysis",
            "validation_criteria": ["source_authority", "recency", "technical_accuracy"]
        }
    
    def _create_fallback_synthesis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback synthesis if LLM synthesis fails"""
        return {
            "executive_summary": f"Research completed with {len(results)} sources found.",
            "key_findings": ["Multiple sources identified", "Further analysis recommended"],
            "technical_details": {
                "concepts": [],
                "implementations": [],
                "tools": []
            },
            "recommendations": ["Review source materials", "Conduct deeper analysis"],
            "challenges": [],
            "related_concepts": [],
            "confidence_score": 0.5,
            "knowledge_nodes": [],
            "synthesis_timestamp": datetime.utcnow().isoformat(),
            "source_count": len(results),
            "research_depth": "basic"
        }
    
    async def get_research_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent research history"""
        history = await self.memory_service.retrieve_knowledge(
            'research_synthesis',
            {},
            limit=limit
        )
        return history
    
    async def search_knowledge_graph(self, concept: str) -> List[Dict[str, Any]]:
        """Search knowledge graph for related concepts"""
        related_nodes = await self.memory_service.retrieve_knowledge(
            'knowledge_graph',
            {'concept': concept}
        )
        return related_nodes
```

### Task 5.2: Research Data Models

**File**: `core/models.py` (Enhancement)

**Research Models**:
```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ResearchTask(BaseModel):
    query: str
    research_type: str
    scope: str = "comprehensive"  # basic, moderate, comprehensive
    depth: str = "medium"  # shallow, medium, deep
    time_range: str = "recent"  # recent, all_time, specific_date
    sources: List[str] = ["web", "documentation"]
    filters: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    priority: int = 1
    deadline: Optional[datetime] = None

class SearchResult(BaseModel):
    title: str
    url: str
    content: Optional[str] = None
    snippet: Optional[str] = None
    provider: str
    published_date: Optional[str] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = {}

class KnowledgeNode(BaseModel):
    concept: str
    description: str
    category: str
    relationships: List[str] = []
    confidence: float = 0.0
    sources: List[str] = []
    created_at: datetime
    updated_at: datetime

class ResearchSynthesis(BaseModel):
    session_id: str
    query: str
    research_type: str
    executive_summary: str
    key_findings: List[str]
    technical_details: Dict[str, Any]
    recommendations: List[str]
    challenges: List[Dict[str, str]]
    related_concepts: List[str]
    confidence_score: float
    knowledge_nodes: List[KnowledgeNode]
    source_count: int
    research_depth: str
    synthesis_timestamp: datetime
```

### Task 5.3: Backend API Integration

**File**: `app/api/research.py`

**Research API Endpoints**:
```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/api/research", tags=["research"])

@router.post("/search")
async def conduct_research(research_request: Dict[str, Any], background_tasks: BackgroundTasks):
    """Initiate research task"""
    try:
        research_agent = await get_research_agent()
        
        # Start research in background
        background_tasks.add_task(research_agent.process_task, research_request)
        
        return {
            "status": "research_initiated",
            "message": "Research task started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}")
async def get_research_session(session_id: str):
    """Get research session results"""
    try:
        research_agent = await get_research_agent()
        session = research_agent.active_research_sessions.get(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_research_history(limit: int = 10):
    """Get research history"""
    try:
        research_agent = await get_research_agent()
        history = await research_agent.get_research_history(limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-graph/{concept}")
async def search_knowledge_graph(concept: str):
    """Search knowledge graph for concept"""
    try:
        research_agent = await get_research_agent()
        related_nodes = await research_agent.search_knowledge_graph(concept)
        return related_nodes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quick-search")
async def quick_search(query: str, provider: str = "tavily"):
    """Perform quick search with single provider"""
    try:
        search_service = await get_search_service()
        
        if provider == "tavily":
            results = await search_service.search_tavily(query)
        elif provider == "exa":
            results = await search_service.search_exa(query)
        elif provider == "google":
            results = await search_service.search_google(query)
        else:
            raise HTTPException(status_code=400, detail="Invalid provider")
        
        return {
            "query": query,
            "provider": provider,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_research_agent():
    """Get Research Agent instance"""
    # Implementation to get agent from registry
    pass

async def get_search_service():
    """Get Search Service instance"""
    # Implementation to get search service
    pass
```

### Task 5.4: Frontend Research Interface

**File**: `frontend/components/research-interface.tsx`

**Research Interface Component**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

interface ResearchResult {
  session_id: string;
  research_type: string;
  query: string;
  synthesis: {
    executive_summary: string;
    key_findings: string[];
    recommendations: string[];
    confidence_score: number;
  };
  raw_results: any[];
}

interface ResearchInterfaceProps {
  onResearchComplete: (result: ResearchResult) => void;
}

export const ResearchInterface: React.FC<ResearchInterfaceProps> = ({ onResearchComplete }) => {
  const [query, setQuery] = useState('');
  const [researchType, setResearchType] = useState('technical_research');
  const [scope, setScope] = useState('comprehensive');
  const [isResearching, setIsResearching] = useState(false);
  const [currentSession, setCurrentSession] = useState<string | null>(null);
  const [results, setResults] = useState<ResearchResult | null>(null);
  const [researchHistory, setResearchHistory] = useState<any[]>([]);
  
  useEffect(() => {
    fetchResearchHistory();
  }, []);
  
  const fetchResearchHistory = async () => {
    try {
      const response = await fetch('/api/research/history');
      const history = await response.json();
      setResearchHistory(history);
    } catch (error) {
      console.error('Failed to fetch research history:', error);
    }
  };
  
  const startResearch = async () => {
    if (!query.trim()) return;
    
    setIsResearching(true);
    setResults(null);
    
    try {
      const researchRequest = {
        type: 'research_request',
        query: query,
        research_type: researchType,
        scope: scope,
        depth: 'medium',
        time_range: 'recent',
        sources: ['web', 'documentation']
      };
      
      const response = await fetch('/api/research/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(researchRequest)
      });
      
      if (response.ok) {
        // Poll for results
        pollForResults();
      }
    } catch (error) {
      console.error('Failed to start research:', error);
      setIsResearching(false);
    }
  };
  
  const pollForResults = async () => {
    // Implementation would poll the backend for research completion
    // For now, simulate with timeout
    setTimeout(async () => {
      try {
        // Simulate getting results
        const mockResult: ResearchResult = {
          session_id: 'session_123',
          research_type: researchType,
          query: query,
          synthesis: {
            executive_summary: 'Research completed successfully with comprehensive findings.',
            key_findings: [
              'Key finding 1 from research',
              'Key finding 2 from research',
              'Key finding 3 from research'
            ],
            recommendations: [
              'Recommendation 1',
              'Recommendation 2'
            ],
            confidence_score: 0.85
          },
          raw_results: []
        };
        
        setResults(mockResult);
        setIsResearching(false);
        onResearchComplete(mockResult);
        await fetchResearchHistory();
      } catch (error) {
        console.error('Failed to get research results:', error);
        setIsResearching(false);
      }
    }, 5000);
  };
  
  const quickSearch = async (provider: string) => {
    if (!query.trim()) return;
    
    try {
      const response = await fetch(`/api/research/quick-search?query=${encodeURIComponent(query)}&provider=${provider}`, {
        method: 'POST'
      });
      
      const data = await response.json();
      console.log(`${provider} results:`, data);
    } catch (error) {
      console.error(`Failed to search with ${provider}:`, error);
    }
  };
  
  return (
    <div className="research-interface space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Research Assistant</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Research Query</label>
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your research question or topic..."
              className="min-h-[100px]"
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Research Type</label>
              <Select value={researchType} onValueChange={setResearchType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="technical_research">Technical Research</SelectItem>
                  <SelectItem value="market_research">Market Research</SelectItem>
                  <SelectItem value="competitive_analysis">Competitive Analysis</SelectItem>
                  <SelectItem value="documentation_search">Documentation Search</SelectItem>
                  <SelectItem value="best_practices">Best Practices</SelectItem>
                  <SelectItem value="troubleshooting">Troubleshooting</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium">Research Scope</label>
              <Select value={scope} onValueChange={setScope}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="basic">Basic</SelectItem>
                  <SelectItem value="moderate">Moderate</SelectItem>
                  <SelectItem value="comprehensive">Comprehensive</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <div className="flex gap-2">
            <Button 
              onClick={startResearch} 
              disabled={isResearching || !query.trim()}
              className="flex-1"
            >
              {isResearching ? 'Researching...' : 'Start Research'}
            </Button>
            
            <Button 
              variant="outline" 
              onClick={() => quickSearch('tavily')}
              disabled={!query.trim()}
            >
              Quick Tavily
            </Button>
            
            <Button 
              variant="outline" 
              onClick={() => quickSearch('exa')}
              disabled={!query.trim()}
            >
              Quick Exa
            </Button>
          </div>
        </CardContent>
      </Card>
      
      {results && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Research Results</span>
              <Badge variant="outline">
                Confidence: {Math.round(results.synthesis.confidence_score * 100)}%
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="summary" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="summary">Summary</TabsTrigger>
                <TabsTrigger value="findings">Key Findings</TabsTrigger>
                <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
              </TabsList>
              
              <TabsContent value="summary" className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Executive Summary</h4>
                  <p className="text-sm text-gray-600">{results.synthesis.executive_summary}</p>
                </div>
              </TabsContent>
              
              <TabsContent value="findings" className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Key Findings</h4>
                  <ul className="space-y-2">
                    {results.synthesis.key_findings.map((finding, index) => (
                      <li key={index} className="text-sm text-gray-600 flex items-start">
                        <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                        {finding}
                      </li>
                    ))}
                  </ul>
                </div>
              </TabsContent>
              
              <TabsContent value="recommendations" className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Recommendations</h4>
                  <ul className="space-y-2">
                    {results.synthesis.recommendations.map((rec, index) => (
                      <li key={index} className="text-sm text-gray-600 flex items-start">
                        <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
      
      {researchHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Research</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {researchHistory.slice(0, 5).map((item, index) => (
                <div key={index} className="flex items-center justify-between p-2 border rounded">
                  <div>
                    <div className="text-sm font-medium">{item.query}</div>
                    <div className="text-xs text-gray-500">{item.research_type}</div>
                  </div>
                  <Badge variant="outline">{item.research_depth}</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
```

## Testing Strategy

### Task 5.5: Research Agent Testing

**Unit Tests**:
```python
# test_research_agent.py
class TestResearchAgent:
    async def test_research_task_processing(self):
        pass
    
    async def test_multi_provider_search(self):
        pass
    
    async def test_result_synthesis(self):
        pass
    
    async def test_knowledge_storage(self):
        pass
    
    async def test_research_planning(self):
        pass
```

**Integration Tests**:
- End-to-end research workflow
- Search provider integration
- Knowledge graph construction
- Frontend research interface

## Validation Criteria

### Backend Validation
- [ ] Research agent processes queries correctly
- [ ] Multi-provider search integration functional
- [ ] Result synthesis produces quality output
- [ ] Knowledge storage and retrieval working
- [ ] API endpoints respond correctly

### Frontend Validation
- [ ] Research interface accepts user input
- [ ] Real-time research progress display
- [ ] Results visualization comprehensive
- [ ] Research history accessible
- [ ] Quick search functions work

### Integration Validation
- [ ] Complete research workflow execution
- [ ] Knowledge persistence across sessions
- [ ] Frontend reflects backend research state
- [ ] Error handling for failed searches
- [ ] Performance with large result sets

## Human Testing Scenarios

1. **Basic Research Test**: Conduct simple technical research query
2. **Comprehensive Research Test**: Execute complex multi-faceted research
3. **Provider Comparison Test**: Compare results from different search providers
4. **Knowledge Graph Test**: Verify knowledge node creation and relationships
5. **Research History Test**: Access and review previous research sessions
6. **Error Handling Test**: Test with invalid queries and network issues

## Next Steps

After successful validation of the Research Agent, proceed to **06-architect-planner-agent-enhancement.md** for implementing the enhanced Architect Planner Agent with advanced system design capabilities.

---

**Dependencies**: This phase requires the search service from Phase 2, agent framework from Phase 3, and memory service integration.