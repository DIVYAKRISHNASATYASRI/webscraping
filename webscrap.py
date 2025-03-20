import asyncio
import json
import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Union, Tuple
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, LLMExtractionStrategy
from urllib.parse import urlparse, quote
from googlesearch import search, SearchResult  # Import SearchResult
import logging
import hashlib

# Load environment variables (if any)
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pydantic Models
class FinalDataSchema(BaseModel):
    document_url: str = Field(..., description="Official document URL")
    relevant_information: Optional[str] = Field(None, description="Information relevant to the user's query")
    confidence_score: Optional[float] = Field(None, description="Confidence score (0-1) of the extracted information")
    extracted_date: Optional[str] = Field(None, description="Date when the information was extracted")
    keywords_found: Optional[List[str]] = Field(None, description="List of keywords found related to the query")

class SummarySchema(BaseModel):
    summary: str = Field(..., description="A concise summary of the extracted information")


# Get LLM extraction strategy
def get_llm_strategy(user_prompt: str) -> LLMExtractionStrategy:
    instruction = f"""
    You are an expert information extractor. Your task is to extract information relevant to the user's request from a webpage. 
    The user's request is: '{user_prompt}'. 
    Focus on extracting only information directly answering the user's question.
    Provide the answer in the 'relevant_information' field.
    Include the source document URL in the 'document_url' field.
    Provide a confidence score (0-1) indicating the certainty of the extracted information.
    Include today's date (YYYY-MM-DD format) in the 'extracted_date' field.
    Provide keywords found in the webpage related to the user's query in 'keywords_found' field.
    If no relevant information is found, leave 'relevant_information' blank.
    """

    simplified_schema = {
        "type": "object",
        "properties": {
            "document_url": {"type": "string", "description": "Official document URL"},
            "relevant_information": {"type": "string", "description": "Information relevant to the user's query"},
            "confidence_score": {"type": "number", "format": "float", "description": "Confidence score (0-1)"},
            "extracted_date": {"type": "string", "description": "Date when the information was extracted"},
            "keywords_found": {"type": "array", "items": {"type": "string"}, "description": "List of keywords found"}
        },
        "required": ["document_url", "relevant_information", "confidence_score", "extracted_date", "keywords_found"]
    }

    return LLMExtractionStrategy(
        provider="openai/gpt-4o",
        api_token=os.getenv("OPENAI_API_KEY"),
        schema=simplified_schema,
        extraction_type="schema",
        instruction=instruction,
        input_format="markdown",
        verbose=True,
    )

# Get LLM summarization strategy
def get_llm_summarization_strategy(user_prompt: str, extracted_information: List[Dict[str, Union[str, float, List[str]]]]) -> LLMExtractionStrategy:
    information_string = "\n".join([f"Source: {item['source_url']}\nInformation: {item['information']}" for item in extracted_information])

    instruction = f"""
    You are an expert summarizer. You will be given information extracted from several web pages related to the user's request.
    The user's request was: '{user_prompt}'.
    Your task is to create a concise summary of the following information:

    {information_string}

    Focus on providing a single, coherent summary that answers the user's question, and avoid including URLs in the summary.
    """

    summary_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "A concise summary of the extracted information"}
        },
        "required": ["summary"]
    }

    return LLMExtractionStrategy(
        provider="openai/gpt-4o",
        api_token=os.getenv("OPENAI_API_KEY"),
        schema=summary_schema,
        extraction_type="schema",
        instruction=instruction,
        input_format="markdown",
        verbose=True,
    )

# Enhanced Fetch and process the data (for information extraction)
async def fetch_and_process_data(crawler: AsyncWebCrawler, url: str, llm_strategy: LLMExtractionStrategy, session_id: str) -> Tuple[List[dict], bool]:
    try:
        logging.info(f"Fetching data from: {url}")
        try:
            result = await crawler.arun(
                url=url,  # url is now a string
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    extraction_strategy=llm_strategy,
                    css_selector="",
                    session_id=session_id,
                ),
            )
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request Exception for {url}: {e}")
            return [], False
        except Exception as e:
            logging.exception(f"Exception during crawler.arun for {url}: {e}")
            return [], False


        if not (result.success and result.extracted_content):
            logging.warning(f"Error fetching data from {url}: {result.error_message}")
            return [], False

        extracted_data = json.loads(result.extracted_content)
        if not extracted_data:
            logging.warning(f"No data found on {url}")
            return [], False

        # Validate extracted data against the schema
        validated_data = []
        for item in extracted_data:
            try:
                FinalDataSchema(**item)  # Validate against Pydantic model
                validated_data.append(item)
            except Exception as e:
                logging.error(f"Validation error for item: {item}. Error: {e}")

        return validated_data, False
    except Exception as e:
        logging.exception(f"Exception while fetching data from {url}: {e}")
        return [], True

# NEW: Fetch and process data for summarization
async def fetch_and_process_summary(crawler: AsyncWebCrawler, url: str, llm_strategy: LLMExtractionStrategy, session_id: str) -> Tuple[List[dict], bool]:
    try:
        logging.info(f"Fetching summary from: {url}")
        try:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    extraction_strategy=llm_strategy,
                    css_selector="",
                    session_id=session_id,
                ),
            )
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request Exception for {url}: {e}")
            return [], False
        except Exception as e:
            logging.exception(f"Exception during crawler.arun for {url}: {e}")
            return [], False


        if not (result.success and result.extracted_content):
            logging.warning(f"Error fetching summary: {result.error_message}")
            return [], False

        extracted_data = json.loads(result.extracted_content)
        if not extracted_data:
            logging.warning(f"No summary found.")
            return [], False

        # Validate extracted data against the SummarySchema
        validated_data = []
        for item in extracted_data:
            try:
                SummarySchema(**item)  # Validate against Pydantic model
                validated_data.append(item)
            except Exception as e:
                logging.error(f"Validation error for summary item: {item}. Error: {e}")

        return validated_data, False
    except Exception as e:
        logging.exception(f"Exception while fetching summary: {e}")
        return [], True

#  Scraping with enhanced error handling and logging
async def scrape_and_extract(user_prompt: str, urls: List[str]) -> List[Dict[str, Union[str, float, List[str]]]]:
    llm_strategy = get_llm_strategy(user_prompt)
    session_id = "colab_session_123"
    extracted_info = []
    seen_information = set()  # Track information already processed

    for url in urls:
        try:
            async with AsyncWebCrawler() as crawler:
                data, no_results_found = await fetch_and_process_data(crawler, url, llm_strategy, session_id)
                if no_results_found:
                    logging.warning(f"No data found for URL: {url}")
                elif data:
                    for item in data:
                        if item.get('relevant_information'):
                            info_hash = hashlib.md5(item['relevant_information'].encode('utf-8')).hexdigest() # De-duplication based on the content
                            if info_hash not in seen_information:
                                extracted_info.append({
                                    "information": item['relevant_information'],
                                    "source_url": item['document_url'],
                                    "confidence_score": item.get('confidence_score'),
                                    "extracted_date": item.get('extracted_date'),
                                    "keywords_found": item.get('keywords_found')
                                })
                                seen_information.add(info_hash)  # Mark the information as processed
                            else:
                                logging.info(f"Skipping duplicate information from {url}")

        except Exception as e:
            logging.exception(f"Exception in scrape_and_extract for URL {url}: {e}")
    return extracted_info

#  URL generation with site limiting
async def generate_urls_and_scrape(user_prompt: str, site: str = None, num_results: int = 3) -> List[Dict[str, Union[str, float, List[str]]]]:
    query = user_prompt
    if site:
        query += f" site:{site}"

    try:
        urls: List[SearchResult] = list(search(query, num_results=num_results, advanced=True))
        # Extract URL strings from SearchResult objects
        url_strings = [result.url for result in urls]
    except Exception as e:
        logging.error(f"Error during Google Search: {e}")
        return []

    extracted_info = await scrape_and_extract(user_prompt, url_strings)  # Pass the list of string URLs
    return extracted_info

#  Summarize extracted information
async def summarize_information(user_prompt: str, extracted_information: List[Dict[str, Union[str, float, List[str]]]]) -> str:
    if not extracted_information:
        return "No relevant information found to summarize."

    llm_strategy = get_llm_summarization_strategy(user_prompt, extracted_information)
    session_id = "colab_session_123_summary" #Different session id for summarization

    try:
        async with AsyncWebCrawler() as crawler:
            summary_data, no_results_found = await fetch_and_process_summary(crawler, "https://www.example.com", llm_strategy, session_id)  #Dummy URL needed

            if no_results_found or not summary_data:
                return "Unable to generate summary."

            return summary_data[0]['summary']  # Assuming first element contains summary
    except Exception as e:
        logging.exception(f"Error during summarization: {e}")
        return "Error generating summary."

#  Main function with exception handling and user-friendly output
if __name__ == "__main__":
    try:
        user_prompt = input("Enter your query: ")
        extracted_data = asyncio.run(generate_urls_and_scrape(user_prompt))

        if extracted_data:
            logging.info("Extracted data successfully.")  # Log here

            #Summarize the information
            summary = asyncio.run(summarize_information(user_prompt, extracted_data))
            print("\n--- Summary ---")
            print(summary)

            #List Sources
            print("\n--- Sources ---")
            for item in extracted_data:
                print(f"- {item['source_url']}")
        else:
            print("No relevant information found.")
            logging.info("No relevant data found.")# Log here

    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {e}")
        print(f"An unexpected error occurred: {e}. Check the logs for details.")