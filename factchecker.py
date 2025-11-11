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
        
        # Define tool schema for function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for current information to verify facts.",
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
        """Perform web search using Tavily API."""
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
        """Verify factual accuracy of the text using AI + web search."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert fact-checker. 
                    1. Use the search_web tool when you need to verify claims. 
                    2. Evaluate multiple sources. 
                    3. Give a verdict (TRUE, FALSE, UNVERIFIABLE, ERROR). 
                    4. Confidence (0.0–1.0). 
                    5. Write a summary, reasoning, and cite sources."""
                },
                {
                    "role": "user",
                    "content": f"Fact-check this claim:\n\n\"{text}\""
                }
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            messages.append(response_message)
            
            # Handle tool calls
            search_results_data = []
            while getattr(response_message, "tool_calls", None):
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "search_web":
                        args = json.loads(tool_call.function.arguments)
                        query = args.get("query")
                        print(f"🔍 Searching web for: {query}")
                        
                        results = self.search_web(query)
                        search_results_data.append(results)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": json.dumps(results)
                        })

                # Ask model to process tool output
                response = self.openai_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
                response_message = response.choices[0].message
                messages.append(response_message)
            
            # Final structured JSON output
            final_prompt = """Now summarize your findings in JSON with:
            - verdict: TRUE, FALSE, UNVERIFIABLE, or ERROR
            - confidence: number 0.0–1.0
            - summary: 1–2 sentences
            - reasoning: detailed explanation
            - sources: [{title, url}]"""
            
            messages.append({"role": "user", "content": final_prompt})
            
            final_response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=messages
            )
            
            result_data = json.loads(final_response.choices[0].message.content)
            
            sources = [Source(title=s.get("title", ""), url=s.get("url", "")) for s in result_data.get("sources", [])]
            
            return CheckResponse(
                verdict=result_data.get("verdict", "ERROR"),
                confidence=float(result_data.get("confidence", 0.0)),
                summary=result_data.get("summary", "Unable to verify"),
                reasoning=result_data.get("reasoning", ""),
                sources=sources,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            print(f"Error in fact-checking: {str(e)}")
            return CheckResponse(
                verdict="ERROR",
                confidence=0.0,
                summary=f"Error during fact-checking: {str(e)}",
                reasoning="",
                sources=[],
                timestamp=datetime.utcnow()
            )
