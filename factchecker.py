from openai import OpenAI
from tavily import TavilyClient
from typing import List, Dict
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from models import CheckResponse, Source
import base64
import PyPDF2

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

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from an image using GPT-4V vision capabilities.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
        """
        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
            
            # Determine image format from file extension
            file_ext = os.path.splitext(image_path)[1].lower()
            mime_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            mime_type = mime_type_map.get(file_ext, "image/jpeg")
            
            # Use GPT-4V to extract text
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please extract and transcribe all text visible in this image. Return only the extracted text."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            extracted_text = response.choices[0].message.content
            print(f"✅ Text extracted from image: {extracted_text[:100]}...")
            return extracted_text
            
        except Exception as e:
            print(f"Error extracting text from image: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        try:
            extracted_text = ""
            
            with open(pdf_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                
                print(f"📄 Extracting text from {num_pages} pages...")
                
                # Extract text from each page
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    extracted_text += page.extract_text()
                    print(f"   Page {page_num + 1}/{num_pages} extracted")
            
            print(f"✅ Text extracted from PDF: {extracted_text[:100]}...")
            return extracted_text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise

    def check_image(self, image_path: str) -> CheckResponse:
        """
        Fact-check text extracted from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            CheckResponse with fact-check verdict
        """
        try:
            # Extract text from the image
            extracted_text = self.extract_text_from_image(image_path)
            
            if not extracted_text.strip():
                return CheckResponse(
                    verdict="UNVERIFIABLE",
                    confidence=0.0,
                    summary="No text found in the image to fact-check.",
                    reasoning="",
                    sources=[],
                    timestamp=datetime.utcnow()
                )
            
            # Fact-check the extracted text
            return self.check_text(extracted_text)
            
        except Exception as e:
            print(f"Error fact-checking image: {str(e)}")
            return CheckResponse(
                verdict="ERROR",
                confidence=0.0,
                summary=f"Error processing image: {str(e)}",
                reasoning="",
                sources=[],
                timestamp=datetime.utcnow()
            )

    def check_pdf(self, pdf_path: str) -> CheckResponse:
        """
        Fact-check text extracted from a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            CheckResponse with fact-check verdict
        """
        try:
            # Extract text from the PDF
            extracted_text = self.extract_text_from_pdf(pdf_path)
            
            if not extracted_text.strip():
                return CheckResponse(
                    verdict="UNVERIFIABLE",
                    confidence=0.0,
                    summary="No text found in the PDF to fact-check.",
                    reasoning="",
                    sources=[],
                    timestamp=datetime.utcnow()
                )
            
            # Fact-check the extracted text
            return self.check_text(extracted_text)
            
        except Exception as e:
            print(f"Error fact-checking PDF: {str(e)}")
            return CheckResponse(
                verdict="ERROR",
                confidence=0.0,
                summary=f"Error processing PDF: {str(e)}",
                reasoning="",
                sources=[],
                timestamp=datetime.utcnow()
            )

    def check_url(self, url: str) -> CheckResponse:
        pass
    