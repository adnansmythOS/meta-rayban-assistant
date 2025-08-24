# Enhanced Multi-Agent Chatbot with Streaming Support

import os
from dotenv import load_dotenv
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import re
from urllib.parse import urlparse

# Core dependencies
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# LangChain components
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser

# Web scraping and search
from playwright.async_api import async_playwright
from tavily import TavilyClient
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MODEL_NAME = "claude-3-haiku-20240307"
    MAX_SEARCH_RESULTS = 10
    MAX_SCRAPE_PAGES = 3
    CONVERSATION_WINDOW_SIZE = 10
    SESSION_TIMEOUT_HOURS = 24
    CONFIDENCE_THRESHOLD = 0.7

# Data models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    clear_history: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str
    timestamp: str
    status: str
    conversation_length: int
    context_used: bool
    debug_info: Optional[Dict[str, Any]] = None

# Session Management
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.last_cleanup = datetime.now()
    
    def cleanup_expired_sessions(self):
        if datetime.now() - self.last_cleanup > timedelta(hours=1):
            current_time = datetime.now()
            expired_sessions = [sid for sid, data in self.sessions.items() if current_time - datetime.fromisoformat(data['last_activity']) > timedelta(hours=Config.SESSION_TIMEOUT_HOURS)]
            for session_id in expired_sessions:
                del self.sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")
            self.last_cleanup = current_time
    
    def get_or_create_session(self, session_id: str) -> Dict:
        self.cleanup_expired_sessions()
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'messages': [], 
                'created_at': datetime.now().isoformat(), 
                'last_activity': datetime.now().isoformat(), 
                'message_count': 0
            }
            logger.info(f"Created new session: {session_id}")
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str, sources: List[str] = None):
        session = self.get_or_create_session(session_id)
        session['messages'].append({
            'role': role, 
            'content': content, 
            'timestamp': datetime.now().isoformat(), 
            'sources': sources or []
        })
        session['last_activity'] = datetime.now().isoformat()
        session['message_count'] += 1
        if len(session['messages']) > Config.CONVERSATION_WINDOW_SIZE * 2:
            session['messages'] = session['messages'][-Config.CONVERSATION_WINDOW_SIZE * 2:]
            logger.info(f"Trimmed conversation history for session: {session_id}")
    
    def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        session = self.get_or_create_session(session_id)
        return [
            HumanMessage(content=msg['content']) if msg['role'] == 'user' 
            else AIMessage(content=msg['content']) 
            for msg in session['messages']
        ]
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].update({'messages': [], 'message_count': 0})
            logger.info(f"Cleared session history: {session_id}")

# Score-based Query Router Agent (Guardrail)
class QueryRouterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a strict classification agent. Your task is to determine if a user's query is related to "Meta Ray-Ban smart glasses" and provide a confidence score.

The query must be about the product's features, price, models, comparisons, troubleshooting, software, or accessories.

You MUST respond with ONLY a valid JSON object with exactly two keys:
1. "is_relevant": boolean (true or false)
2. "confidence": number between 0.0 and 1.0

Examples of RELEVANT queries (high confidence):
- "What's the battery life of Ray-Ban smart glasses?"
- "Compare Wayfarer and Headliner models"
- "How do I connect my Ray-Ban glasses to my phone?"
- "What's the price of Meta Ray-Ban glasses?"

Examples of IRRELEVANT queries (should be false):
- "What's the weather today?"
- "Who is the CEO of Meta?"
- "How to cook pasta?"
- "Tell me a joke"

Your response must be ONLY the JSON object, nothing else."""),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def classify_query(self, query: str) -> Dict[str, Any]:
        try:
            logger.info(f"Classifying query: {query}")
            result = await self.chain.ainvoke({"query": query})
            
            cleaned_result = result.strip()
            
            json_match = re.search(r'\{[^}]*\}', cleaned_result)
            if json_match:
                json_str = json_match.group()
                classification = json.loads(json_str)
            else:
                classification = json.loads(cleaned_result)
            
            if "is_relevant" in classification and "confidence" in classification:
                is_relevant = bool(classification["is_relevant"])
                confidence = float(classification["confidence"])
                
                logger.info(f"Classification result: relevant={is_relevant}, confidence={confidence}")
                return {"is_relevant": is_relevant, "confidence": confidence}
            else:
                logger.warning(f"Invalid classification structure: {classification}")
                return {"is_relevant": False, "confidence": 0.0}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in query classification: {str(e)}. Raw result: {result}")
            return {"is_relevant": False, "confidence": 0.0}
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}. Defaulting to NOT RELEVANT.")
            return {"is_relevant": False, "confidence": 0.0}

# Research Planner Agent
class ContextualResearchPlannerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Research Planner Agent specializing in Meta Ray-Ban smart glasses queries. Your goal is to create a research plan that prioritizes official and primary sources.

Given a user query, create a research plan.

IMPORTANT:
1. You MUST prioritize official sources.
2. For at least one of your generated queries, use the `site:` search operator to look specifically within `ray-ban.com` or `meta.com`.
3. For another query, include keywords like "official product page", "specs", or "documentation".

You MUST respond with ONLY a valid JSON object with exactly these keys:
{
    "search_queries": ["query1", "query2", "query3"],
    "key_topics": ["topic1", "topic2"],
    "priority": "high"
}

Examples:
Query: "What's the battery life?" 
Response: {
    "search_queries": [
        "Meta Ray-Ban smart glasses battery life site:ray-ban.com", 
        "official documentation Ray-Ban Stories battery duration",
        "Meta smart glasses battery performance review"
    ], 
    "key_topics": ["battery", "specifications", "charging"], 
    "priority": "high"
}

Query: "What are the different models available?"
Response": {
    "search_queries": [
        "official Meta Ray-Ban smart glasses models page site:ray-ban.com",
        "compare Ray-Ban smart glasses Wayfarer vs Headliner",
        "all available Meta smart glasses styles"
    ],
    "key_topics": ["models", "styles", "Wayfarer", "Headliner"],
    "priority": "high"
}
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def plan_research(self, query: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        try:
            logger.info(f"Planning research for query: {query}")
            result = await self.chain.ainvoke({"query": query, "chat_history": chat_history})
            
            cleaned_result = result.strip()
            json_match = re.search(r'\{.*\}', cleaned_result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                plan = json.loads(json_str)
            else:
                plan = json.loads(cleaned_result)
            
            required_keys = ["search_queries", "key_topics", "priority"]
            if all(key in plan for key in required_keys):
                logger.info(f"Research plan: {plan}")
                return plan
            else:
                logger.warning(f"Invalid plan structure: {plan}")
                return self._get_fallback_plan(query)
                
        except Exception as e:
            logger.error(f"Error in contextual research planning: {str(e)}")
            return self._get_fallback_plan(query)
    
    def _get_fallback_plan(self, query: str) -> Dict[str, Any]:
        return {
            "search_queries": [
                f"Meta Ray-Ban smart glasses {query} site:ray-ban.com", 
                f"Ray-Ban Stories {query}"
            ],
            "key_topics": ["features", "specifications"],
            "priority": "medium"
        }

# Synthesizer Agent with Streaming Support (Fixed - No Sources in Answer)
class ContextualSynthesizerCiterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Synthesizer Agent specializing in Meta Ray-Ban smart glasses information.
Your role is to analyze raw web data and conversation history to create coherent, contextual answers.

IMPORTANT: Provide ONLY the answer content. Do NOT include source URLs or references in your response. The sources will be handled separately by the system.

Your response should be:
- A comprehensive, well-structured answer based on the provided web content
- Contextual and relevant to the user's query
- Clear and informative
- WITHOUT any source citations, URLs, or reference mentions

Focus solely on synthesizing the information into a helpful answer."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Current Query: {query}\n\nNew Web Content:\n{web_content}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def synthesize_answer_streaming(self, query: str, web_data: List[Dict[str, Any]], chat_history: List[BaseMessage]):
        """Generator function for streaming response"""
        try:
            web_content = "\n".join([
                f"\n--- Source: {item.get('title', '')} ({item.get('url', '')}) ---\n{item.get('content', item.get('snippet', ''))}" 
                for item in web_data if item.get("url") and item.get("content", item.get("snippet"))
            ])
            
            if not web_content.strip():
                yield json.dumps({
                    "type": "sentence",
                    "content": "I apologize, but I couldn't find specific information about that topic. Could you please rephrase your question or ask about a different aspect of Meta Ray-Ban smart glasses?"
                }) + "\n"
                return
            
            sources = [item['url'] for item in web_data if item.get('url')]
            
            # Format the messages for streaming LLM
            messages = self.prompt.format_messages(
                query=query,
                web_content=web_content,
                chat_history=chat_history
            )
            
            # Stream directly from the LLM with formatted messages
            accumulated_content = ""
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    accumulated_content += chunk.content
                    
                    # Check if we have complete sentences
                    sentences = self._extract_complete_sentences(accumulated_content)
                    if sentences["complete_sentences"]:
                        # Send complete sentences
                        for sentence in sentences["complete_sentences"]:
                            if sentence.strip():
                                yield json.dumps({
                                    "type": "sentence",
                                    "content": sentence.strip()
                                }) + "\n"
                        
                        # Update accumulated content to remaining partial sentence
                        accumulated_content = sentences["remaining"]
            
            # Send any remaining content
            if accumulated_content.strip():
                yield json.dumps({
                    "type": "sentence",
                    "content": accumulated_content.strip()
                }) + "\n"
            
            # Send sources at the end
            yield json.dumps({
                "type": "sources",
                "content": list(set(sources))
            }) + "\n"
            
            # Send completion signal
            yield json.dumps({
                "type": "complete",
                "timestamp": datetime.now().isoformat(),
                "context_used": len(chat_history) > 0
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {str(e)}")
            context_note = " I'm having trouble accessing my previous responses, but I'll do my best." if chat_history else ""
            yield json.dumps({
                "type": "error",
                "content": f"I apologize, but I encountered an error processing your request.{context_note} Please try rephrasing your question about Meta Ray-Ban smart glasses."
            }) + "\n"
    
    def _extract_complete_sentences(self, text: str) -> Dict[str, Any]:
        """Extract complete sentences from accumulated text"""
        import re
        
        # Split text into sentences, keeping the delimiters
        sentence_pattern = r'([.!?]+)'
        parts = re.split(sentence_pattern, text)
        
        complete_sentences = []
        remaining = ""
        
        i = 0
        while i < len(parts) - 1:  # Don't include the last part if it doesn't end with punctuation
            sentence_content = parts[i]
            if i + 1 < len(parts):
                punctuation = parts[i + 1]
                complete_sentence = sentence_content + punctuation
                complete_sentences.append(complete_sentence)
                i += 2
            else:
                i += 1
        
        # The remaining part (incomplete sentence)
        if parts:
            remaining = parts[-1] if not re.match(sentence_pattern, parts[-1]) else ""
        
        return {
            "complete_sentences": complete_sentences,
            "remaining": remaining
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming - DEPRECATED, use _extract_complete_sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]

    async def synthesize_answer(self, query: str, web_data: List[Dict[str, Any]], chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """Non-streaming version for backward compatibility"""
        try:
            web_content = "\n".join([
                f"\n--- Source: {item.get('title', '')} ({item.get('url', '')}) ---\n{item.get('content', item.get('snippet', ''))}" 
                for item in web_data if item.get("url") and item.get("content", item.get("snippet"))
            ])
            
            if not web_content.strip():
                return {
                    "answer": "I apologize, but I couldn't find specific information about that topic. Could you please rephrase your question or ask about a different aspect of Meta Ray-Ban smart glasses?",
                    "sources": [],
                    "timestamp": datetime.now().isoformat(),
                    "context_used": len(chat_history) > 0
                }
            
            sources = [item['url'] for item in web_data if item.get('url')]
            result = await self.chain.ainvoke({
                "query": query, 
                "web_content": web_content, 
                "chat_history": chat_history
            })
            
            return {
                "answer": result,
                "sources": list(set(sources)),
                "timestamp": datetime.now().isoformat(),
                "context_used": len(chat_history) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in contextual synthesis: {str(e)}")
            context_note = " I'm having trouble accessing my previous responses, but I'll do my best." if chat_history else ""
            return {
                "answer": f"I apologize, but I encountered an error processing your request.{context_note} Please try rephrasing your question about Meta Ray-Ban smart glasses.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "context_used": False
            }

# Main Orchestrator with Streaming Support
class ContextualMultiAgentOrchestrator:
    def __init__(self):
        self.llm = ChatAnthropic(
            model=Config.MODEL_NAME, 
            anthropic_api_key=Config.ANTHROPIC_API_KEY, 
            temperature=0.1,
            streaming=True  # Enable streaming
        )
        self.router = QueryRouterAgent(self.llm)
        self.planner = ContextualResearchPlannerAgent(self.llm)
        self.searcher = WebSearchScrapeAgent()
        self.synthesizer = ContextualSynthesizerCiterAgent(self.llm)
        self.session_manager = SessionManager()
    
    def _create_denial_response(self, session_id: str, debug_info: Dict[str, Any]) -> Dict[str, Any]:
        session = self.session_manager.get_or_create_session(session_id)
        return {
            "answer": "I am a specialized assistant for Meta Ray-Ban smart glasses. I can only answer questions about these smart glasses, their features, specifications, pricing, models, troubleshooting, and accessories. Please ask me something about Meta Ray-Ban smart glasses.",
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "context_used": False,
            "conversation_length": len(session["messages"]) // 2,
            "debug_info": debug_info
        }
    
    async def process_query_streaming(self, query: str, session_id: str, clear_history: bool = False):
        """Streaming version of process_query"""
        try:
            if clear_history:
                self.session_manager.clear_session(session_id)
            
            chat_history = self.session_manager.get_conversation_history(session_id)
            logger.info(f"Processing query for session {session_id}: {query}")
            
            debug_info = {}

            # Stage 1: Query Classification
            logger.info("Stage 1: Classifying query relevance")
            classification = await self.router.classify_query(query)
            debug_info["classification"] = classification
            
            is_relevant = classification.get("is_relevant", False)
            confidence = classification.get("confidence", 0.0)
            
            if not is_relevant or confidence < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"DENIED: Query not relevant or low confidence (Score: {confidence})")
                self.session_manager.add_message(session_id, "user", query)
                denial_response = self._create_denial_response(session_id, debug_info)
                self.session_manager.add_message(session_id, "assistant", denial_response["answer"], [])
                
                yield json.dumps({
                    "type": "sentence",
                    "content": denial_response["answer"]
                }) + "\n"
                yield json.dumps({
                    "type": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "context_used": False,
                    "debug_info": debug_info
                }) + "\n"
                return
            
            # Send status updates
            yield json.dumps({"type": "status", "content": "Planning research..."}) + "\n"
            
            # Stage 2: Research Planning
            logger.info("Stage 2: Planning research")
            research_plan = await self.planner.plan_research(query, chat_history)
            debug_info["research_plan"] = research_plan
            
            if research_plan.get("priority") != "high":
                logger.info(f"DENIED: Query priority not high (Priority: {research_plan.get('priority')})")
                self.session_manager.add_message(session_id, "user", query)
                denial_response = self._create_denial_response(session_id, debug_info)
                self.session_manager.add_message(session_id, "assistant", denial_response["answer"], [])
                
                yield json.dumps({
                    "type": "sentence",
                    "content": denial_response["answer"]
                }) + "\n"
                yield json.dumps({
                    "type": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "context_used": False,
                    "debug_info": debug_info
                }) + "\n"
                return
            
            # Send status update
            yield json.dumps({"type": "status", "content": "Searching the web..."}) + "\n"
            
            # Stage 3: Web Search and Scraping
            logger.info("Stage 3: Searching and scraping web content")
            web_data = await self.searcher.search_and_scrape(research_plan)
            
            # Send status update
            yield json.dumps({"type": "status", "content": "Generating response..."}) + "\n"
            
            # Stage 4: Streaming Answer Synthesis
            logger.info("Stage 4: Streaming synthesis of final answer")
            
            full_answer = ""
            sources = []
            
            async for chunk in self.synthesizer.synthesize_answer_streaming(query, web_data, chat_history):
                chunk_data = json.loads(chunk.strip())
                
                if chunk_data["type"] == "sentence":
                    full_answer += chunk_data["content"] + " "
                    yield chunk
                elif chunk_data["type"] == "sources":
                    sources = chunk_data["content"]
                    yield chunk
                elif chunk_data["type"] == "complete":
                    chunk_data["debug_info"] = debug_info
                    yield json.dumps(chunk_data) + "\n"
                else:
                    yield chunk
            
            # Save to session
            self.session_manager.add_message(session_id, "user", query)
            self.session_manager.add_message(session_id, "assistant", full_answer.strip(), sources)
            
            logger.info(f"Successfully processed streaming query with {len(sources)} sources")
            
        except Exception as e:
            logger.error(f"Critical error in streaming orchestration: {str(e)}", exc_info=True)
            self.session_manager.add_message(session_id, "user", query)
            error_message = f"I apologize, but I encountered a technical error. Please try asking your question about Meta Ray-Ban smart glasses again."
            self.session_manager.add_message(session_id, "assistant", error_message, [])
            
            yield json.dumps({
                "type": "error",
                "content": error_message
            }) + "\n"

    async def process_query(self, query: str, session_id: str, clear_history: bool = False) -> Dict[str, Any]:
        """Non-streaming version for backward compatibility"""
        try:
            if clear_history:
                self.session_manager.clear_session(session_id)
            
            chat_history = self.session_manager.get_conversation_history(session_id)
            logger.info(f"Processing query for session {session_id}: {query}")
            
            debug_info = {}

            # Stage 1: Query Classification
            logger.info("Stage 1: Classifying query relevance")
            classification = await self.router.classify_query(query)
            debug_info["classification"] = classification
            
            is_relevant = classification.get("is_relevant", False)
            confidence = classification.get("confidence", 0.0)
            
            if not is_relevant or confidence < Config.CONFIDENCE_THRESHOLD:
                logger.info(f"DENIED: Query not relevant or low confidence (Score: {confidence})")
                self.session_manager.add_message(session_id, "user", query)
                denial_response = self._create_denial_response(session_id, debug_info)
                self.session_manager.add_message(session_id, "assistant", denial_response["answer"], [])
                return denial_response
            
            # Stage 2: Research Planning
            logger.info("Stage 2: Planning research")
            research_plan = await self.planner.plan_research(query, chat_history)
            debug_info["research_plan"] = research_plan
            
            if research_plan.get("priority") != "high":
                logger.info(f"DENIED: Query priority not high (Priority: {research_plan.get('priority')})")
                self.session_manager.add_message(session_id, "user", query)
                denial_response = self._create_denial_response(session_id, debug_info)
                self.session_manager.add_message(session_id, "assistant", denial_response["answer"], [])
                return denial_response
            
            # Stage 3: Web Search and Scraping
            logger.info("Stage 3: Searching and scraping web content")
            web_data = await self.searcher.search_and_scrape(research_plan)
            
            # Stage 4: Answer Synthesis
            logger.info("Stage 4: Synthesizing final answer")
            final_result = await self.synthesizer.synthesize_answer(query, web_data, chat_history)
            
            self.session_manager.add_message(session_id, "user", query)
            self.session_manager.add_message(session_id, "assistant", final_result["answer"], final_result["sources"])
            
            session = self.session_manager.get_or_create_session(session_id)
            final_result["conversation_length"] = len(session["messages"]) // 2
            final_result["debug_info"] = debug_info
            
            logger.info(f"Successfully processed query with {len(final_result['sources'])} sources")
            return final_result
            
        except Exception as e:
            logger.error(f"Critical error in orchestration: {str(e)}", exc_info=True)
            self.session_manager.add_message(session_id, "user", query)
            error_response = {
                "answer": f"I apologize, but I encountered a technical error. Please try asking your question about Meta Ray-Ban smart glasses again.",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "context_used": False,
                "conversation_length": 0,
                "debug_info": {"error": str(e)}
            }
            self.session_manager.add_message(session_id, "assistant", error_response["answer"], [])
            return error_response

# Web Search and Scrape Agent
class WebSearchScrapeAgent:
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY) if Config.TAVILY_API_KEY else None
        self.scraper = PlaywrightScrapingTool()
    
    async def search_and_scrape(self, research_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.tavily_client:
            logger.error("Tavily client not initialized - missing API key")
            return []
        
        search_queries = research_plan.get("search_queries", [])
        if not search_queries:
            logger.warning("No search queries provided in research plan")
            return []
        
        tasks = [self._process_query(query) for query in search_queries]
        query_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for result in query_results:
            if isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")
            elif isinstance(result, list):
                results.extend(result)
        
        return results

    async def _process_query(self, query: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Searching for: {query}")
            search_results = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=Config.MAX_SEARCH_RESULTS
            )
            
            all_results = search_results.get("results", [])
            if not all_results:
                logger.warning(f"No search results found for query: {query}")
                return []

            preferred_domains = ["ray-ban.com", "meta.com"]
            official_results = []
            other_results = []

            for result in all_results:
                try:
                    domain = urlparse(result['url']).netloc
                    if any(preferred in domain for preferred in preferred_domains):
                        official_results.append(result)
                    else:
                        other_results.append(result)
                except Exception:
                    other_results.append(result)

            sorted_results = official_results + other_results
            logger.info(f"Re-ranked results. Found {len(official_results)} official sources.")
            
            results_to_scrape = sorted_results[:Config.MAX_SCRAPE_PAGES]
            
            scrape_tasks = [
                self.scraper._arun(result['url']) 
                for result in results_to_scrape 
                if result.get('url')
            ]
            
            if not scrape_tasks:
                logger.warning(f"No URLs to scrape for query: {query}")
                return []
            
            scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)
            
            query_results = []
            for i, result in enumerate(results_to_scrape):
                if i < len(scraped_contents):
                    content = scraped_contents[i]
                    if isinstance(content, Exception):
                        logger.error(f"Scraping failed for {result.get('url')}: {content}")
                        content = result.get('snippet', '')
                    result['content'] = content
                    query_results.append(result)
            
            return query_results
            
        except Exception as e:
            logger.error(f"Error processing search query '{query}': {str(e)}")
            return []

# Web Scraping Tool
class PlaywrightScrapingTool(BaseTool):
    name: str = "playwright_scraper"
    description: str = "Scrape web pages using Playwright for dynamic content"
    
    async def scrape_url(self, url: str) -> str:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = await context.new_page()
                
                await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                content = await page.evaluate("() => document.body.innerText")
                await browser.close()
                
                return content[:5000].strip() if content else ""
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return ""
    
    def _run(self, url: str) -> str:
        return asyncio.run(self.scrape_url(url))
    
    async def _arun(self, url: str) -> str:
        return await self.scrape_url(url)

# FastAPI Application
app = FastAPI(title="Contextual Multi-Agent Meta Ray-Ban Chatbot", version="2.9.0-streaming")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

orchestrator = ContextualMultiAgentOrchestrator()

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Non-streaming endpoint for backward compatibility"""
    try:
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        result = await orchestrator.process_query(
            query=request.query,
            session_id=session_id,
            clear_history=request.clear_history or False
        )
        return ChatResponse(status="success", session_id=session_id, **result)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-stream")
async def ask_question_stream(request: ChatRequest):
    """Streaming endpoint"""
    try:
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        async def generate_stream():
            try:
                # Send session ID first
                yield json.dumps({
                    "type": "session_id",
                    "content": session_id
                }) + "\n"
                
                async for chunk in orchestrator.process_query_streaming(
                    query=request.query,
                    session_id=session_id,
                    clear_history=request.clear_history or False
                ):
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Error in stream generation: {str(e)}", exc_info=True)
                yield json.dumps({
                    "type": "error",
                    "content": f"Stream error: {str(e)}"
                }) + "\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up stream: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Meta Ray-Ban Chatbot is online and ready to help!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    if not Config.ANTHROPIC_API_KEY or not Config.TAVILY_API_KEY:
        raise ValueError("API keys for ANTHROPIC and TAVILY must be set in .env file")
    
    uvicorn.run("main2_debug:app", host="0.0.0.0", port=8002, reload=True, log_level="info")