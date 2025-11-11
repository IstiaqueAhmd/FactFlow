from openai import OpenAI
from tavily import TavilyClient
from typing import List, Dict
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from models import CheckResponse, Source

load_dotenv()


class FactChecker:
    def __init__(self):
        """Initialize the FactChecker with OpenAI and Tavily clients."""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Define the tool for function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for current information to verify facts. Use this when you need to check claims against real-world data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant information on the web",
                            },
                        },
                        "required": ["query"],
                    },
                }
            }
        ]

    def search_web(self, query: str) -> Dict:
        """
        Search the web using Tavily API.
        
        Args:
            query: The search query string
            
        Returns:
            Dictionary containing search results with titles, URLs, and content
        """
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            
            return {
                "results": response.get("results", []),
                "answer": response.get("answer", "")
            }
        except Exception as e:
            print(f"Error searching web: {str(e)}")
            return {"results": [], "answer": "", "error": str(e)}

    def check_text(self, text: str) -> CheckResponse:
        """
        Check the factual accuracy of the provided text using AI and web search.
        
        Args:
            text: The text claim to fact-check
            
        Returns:
            CheckResponse object with verdict, confidence, summary, reasoning, and sources
        """
        try:
            # Initial conversation with the AI
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert fact-checker. Your job is to verify claims using web searches.
                    
When analyzing a claim:
1. Use the search_web function to find current, reliable information
2. Evaluate multiple sources before making a determination
3. Provide a verdict: TRUE, FALSE, UNVERIFIABLE, or ERROR
4. Give a confidence score (0.0 to 1.0) based on evidence quality
5. Write a clear summary and detailed reasoning
6. Always cite your sources

Be thorough and objective."""
                },
                {
                    "role": "user",
                    "content": f"Please fact-check the following claim:\n\n\"{text}\"\n\nUse web search to find current information and provide a detailed fact-check."
                }
            ]
            
            # First API call - AI will likely request a function call
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            messages.append(response_message)
            
            # Handle function calls
            search_results_data = []
            while response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "search_web":
                        # Execute the web search
                        function_args = json.loads(tool_call.function.arguments)
                        search_query = function_args.get("query")
                        
                        print(f"Searching web for: {search_query}")
                        search_results = self.search_web(search_query)
                        search_results_data.append(search_results)
                        
                        # Add function response to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": json.dumps(search_results)
                            }
                        )
                
                # Get next response from AI
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
                response_message = response.choices[0].message
                messages.append(response_message)
            
            # Final response with structured output
            final_prompt = f"""Based on the search results, provide a structured fact-check in JSON format with these exact keys:
- "verdict": one of "TRUE", "FALSE", "UNVERIFIABLE", or "ERROR"
- "confidence": a number between 0.0 and 1.0
- "summary": a brief summary (1-2 sentences)
- "reasoning": detailed reasoning explaining your verdict
- "sources": array of objects with "title" and "url" keys

Return ONLY the JSON object, no other text."""
            
            messages.append({"role": "user", "content": final_prompt})
            
            final_response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            result_text = final_response.choices[0].message.content
            result_data = json.loads(result_text)
            
            # Parse sources
            sources = []
            for source in result_data.get("sources", []):
                sources.append(Source(
                    title=source.get("title", "Unknown"),
                    url=source.get("url", "")
                ))
            
            # Create CheckResponse object
            check_response = CheckResponse(
                verdict=result_data.get("verdict", "ERROR"),
                confidence=float(result_data.get("confidence", 0.0)),
                summary=result_data.get("summary", "Unable to verify"),
                reasoning=result_data.get("reasoning", ""),
                sources=sources,
                timestamp=datetime.utcnow()
            )
            
            return check_response
            
        except Exception as e:
            print(f"Error in fact-checking: {str(e)}")
            # Return error response
            return CheckResponse(
                verdict="ERROR",
                confidence=0.0,
                summary=f"An error occurred during fact-checking: {str(e)}",
                reasoning="",
                sources=[],
                timestamp=datetime.utcnow()
            )

