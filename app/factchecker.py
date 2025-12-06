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
from io import BytesIO

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
                    "content": f"""You are an expert fact-checker. 
                    1. Use the search_web tool when you need to verify claims. 
                    2. Evaluate multiple sources. 
                    3. Give a verdict (TRUE, FALSE, UNVERIFIABLE). 
                    4. Confidence (0.0–1.0).
                    5. Claim summary.
                    6. A brief Conclusion.
                    7. Evidence-based reasoning (Supporting and Counter Evidence).
                    7. Cite the sources.
                    8. Todays date is {datetime.utcnow().date()}"."""
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
            - verdict: TRUE, FALSE or UNVERIFIABLE
            - confidence: number 0.0–1.0
            - claim: the main claim being checked 
            - conclusion: 1–2 sentences
            - reasoning: detailed explanation
            - evidence: {supporting: [], counter: [optional]} 
            - citations: [{title, url}]"""
            
            messages.append({"role": "user", "content": final_prompt})
            
            final_response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=messages
            )
            
            result_data = json.loads(final_response.choices[0].message.content)
            
            sources = [Source(title=s.get("title", ""), url=s.get("url", "")) for s in result_data.get("sources", result_data.get("citations", []))]
            
            return CheckResponse(
                verdict=result_data.get("verdict", "UNVERIFIABLE"),
                confidence=float(result_data.get("confidence", 0.0)),
                claim=result_data.get("claim", ""),
                conclusion=result_data.get("conclusion", "Unable to verify"),
                evidence=result_data.get("evidence", {"supporting": [], "counter": []}),
                sources=sources,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            print(f"Error in fact-checking: {str(e)}")
            return CheckResponse(
                verdict="ERROR",
                confidence=0.0,
                claim="",
                conclusion=f"Error during fact-checking: {str(e)}",
                evidence={"supporting": [], "counter": []},
                sources=[],
                timestamp=datetime.utcnow()
            )

    def extract_text_from_image(self, image_data) -> str:
        """
        Extract text from an image using GPT-4V vision capabilities.
        
        Args:
            image_data: Either bytes of the image or a file path (str)
            
        Returns:
            Extracted text from the image
        """
        try:
            # Handle both bytes and file paths
            if isinstance(image_data, bytes):
                image_bytes = image_data
                # Default to JPEG if we can't determine format from bytes
                mime_type = "image/jpeg"
            else:
                # Read from file path
                with open(image_data, "rb") as image_file:
                    image_bytes = image_file.read()
                # Determine image format from file extension
                file_ext = os.path.splitext(image_data)[1].lower()
                mime_type_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp"
                }
                mime_type = mime_type_map.get(file_ext, "image/jpeg")
            
            # Encode image to base64
            image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
            
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
                                    "url": f"data:{mime_type};base64,{image_b64}"
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

    def extract_text_from_pdf(self, pdf_data) -> str:
        """
        Extract text from a PDF file or bytes.
        
        Args:
            pdf_data: Either bytes of the PDF or a file path (str)
            
        Returns:
            Extracted text from the PDF
        """
        try:
            extracted_text = ""
            
            # Handle both bytes and file paths
            if isinstance(pdf_data, bytes):
                pdf_file = BytesIO(pdf_data)
            else:
                pdf_file = open(pdf_data, "rb")
            
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            print(f"📄 Extracting text from {num_pages} pages...")
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()
                print(f"   Page {page_num + 1}/{num_pages} extracted")
            
            # Close file if it was opened from a path
            if isinstance(pdf_data, str):
                pdf_file.close()
            
            print(f"✅ Text extracted from PDF: {extracted_text[:100]}...")
            return extracted_text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise

    def check_image(self, image_data) -> CheckResponse:
        """
        Fact-check text extracted from an image.
        
        Args:
            image_data: Either bytes of the image or a file path (str)
            
        Returns:
            CheckResponse with fact-check verdict
        """
        try:
            # Extract text from the image
            extracted_text = self.extract_text_from_image(image_data)
            
            if not extracted_text.strip():
                return CheckResponse(
                    verdict="UNVERIFIABLE",
                    confidence=0.0,
                    claim="",
                    conclusion="No text found in the image to fact-check.",
                    evidence={"supporting": [], "counter": []},
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
                claim="",
                conclusion=f"Error processing image: {str(e)}",
                evidence={"supporting": [], "counter": []},
                sources=[],
                timestamp=datetime.utcnow()
            )

    def check_pdf(self, pdf_data) -> CheckResponse:
        """
        Fact-check text extracted from a PDF.
        
        Args:
            pdf_data: Either bytes of the PDF or a file path (str)
            
        Returns:
            CheckResponse with fact-check verdict
        """
        try:
            # Extract text from the PDF
            extracted_text = self.extract_text_from_pdf(pdf_data)
            
            if not extracted_text.strip():
                return CheckResponse(
                    verdict="UNVERIFIABLE",
                    confidence=0.0,
                    claim="",
                    conclusion="No text found in the PDF to fact-check.",
                    evidence={"supporting": [], "counter": []},
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
                claim="",
                conclusion=f"Error processing PDF: {str(e)}",
                evidence={"supporting": [], "counter": []},
                sources=[],
                timestamp=datetime.utcnow()
            )

    def check_url(self, url: str) -> CheckResponse:
        """
        Fact-check content from a URL.
        
        Args:
            url: URL of the webpage to fact-check
            
        Returns:
            CheckResponse with fact-check verdict
        """
        try:
            # Extract content from the URL using Tavily
            print(f"🌐 Fetching content from URL: {url}")
            
            # Use Tavily to extract content from the URL
            response = self.tavily_client.extract(urls=[url])
            
            if not response or not response.get("results"):
                return CheckResponse(
                    verdict="ERROR",
                    confidence=0.0,
                    claim="",
                    conclusion="Unable to fetch content from the provided URL.",
                    evidence={"supporting": [], "counter": []},
                    sources=[Source(title="Source URL", url=url)],
                    timestamp=datetime.utcnow()
                )
            
            # Get the extracted content
            extracted_content = response["results"][0].get("raw_content", "")
            
            if not extracted_content.strip():
                return CheckResponse(
                    verdict="UNVERIFIABLE",
                    confidence=0.0,
                    claim="",
                    conclusion="No text content found at the URL to fact-check.",
                    evidence={"supporting": [], "counter": []},
                    sources=[Source(title="Source URL", url=url)],
                    timestamp=datetime.utcnow()
                )
            
            print(f"✅ Content extracted from URL: {extracted_content[:100]}...")
            
            # Fact-check the extracted content
            return self.check_text(extracted_content)
            
        except Exception as e:
            print(f"Error fact-checking URL: {str(e)}")
            return CheckResponse(
                verdict="ERROR",
                confidence=0.0,
                claim="",
                conclusion=f"Error processing URL: {str(e)}",
                evidence={"supporting": [], "counter": []},
                sources=[Source(title="Source URL", url=url)],
                timestamp=datetime.utcnow()
            )