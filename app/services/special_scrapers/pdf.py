import asyncio
import logging
import os
from pathlib import Path
import httpx
import io
from urllib.parse import urlparse
from typing import Optional

# Use MarkItDown instead of pymupdf4llm
from markitdown import MarkItDown
# import pymupdf4llm # Removed

# Import custom exceptions relative to the app root
from ...core.exceptions import ScrapingError, ConfigurationError

logger = logging.getLogger(__name__)

# Initialize MarkItDown once (can be reused)
# Pass enable_plugins directly
md_converter = MarkItDown(enable_plugins=False)

async def handle_pdf_url(
    url: str, 
    download_pdfs: bool = False, 
    save_dir: str = "downloaded_pdfs",
    max_size_bytes: Optional[int] = None # Add max size parameter
) -> str:
    """
    Downloads a PDF from a URL, checks size, extracts Markdown content using MarkItDown,
    and optionally saves the downloaded PDF.

    Args:
        url: The URL of the PDF document.
        download_pdfs: If True, save the downloaded PDF locally.
        save_dir: The directory to save PDFs if download_pdfs is True.
        max_size_bytes: Optional maximum size in bytes for the PDF. If exceeded, extraction is skipped.

    Returns:
        The extracted Markdown content of the PDF, or an empty string if skipped due to size.

    Raises:
        ScrapingError: If downloading, saving, or parsing the PDF fails.
        httpx.RequestError: If there's an error during the HTTP request.
    """
    logger.info(f"Attempting to handle PDF URL with MarkItDown: {url}")
    pdf_bytes: Optional[bytes] = None

    # 1. Check PDF Size (if max_size_bytes is set)
    if max_size_bytes is not None:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as head_client:
                logger.debug(f"Checking size for PDF: {url}")
                head_response = await head_client.head(url)
                head_response.raise_for_status()
                content_length = head_response.headers.get('content-length')
                if content_length:
                    pdf_size = int(content_length)
                    if pdf_size > max_size_bytes:
                        logger.warning(
                            f"PDF size ({pdf_size / 1024 / 1024:.2f} MB) exceeds limit "
                            f"({max_size_bytes / 1024 / 1024:.2f} MB). Skipping: {url}"
                        )
                        return "" # Return empty string to indicate skipped file
                    else:
                        logger.debug(f"PDF size ({pdf_size / 1024 / 1024:.2f} MB) is within limit for {url}")
                else:
                    logger.warning(f"Could not determine PDF size from headers for {url}. Proceeding with download attempt.")
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error during size check for {url} (Status {e.response.status_code}). Proceeding with download attempt.")
        except httpx.RequestError as e:
            logger.warning(f"Network error during size check for {url}: {e}. Proceeding with download attempt.")
        except Exception as e:
            logger.warning(f"Unexpected error during size check for {url}: {e}. Proceeding with download attempt.")

    # 2. Download PDF bytes into memory
    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client: # Longer timeout for large PDFs
        try:
            logger.debug(f"Starting PDF download from {url}")
            response = await client.get(url)
            response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)
            pdf_bytes = await response.aread()
            if not pdf_bytes:
                raise ScrapingError(f"Downloaded empty file from {url}")
            logger.info(f"Successfully downloaded PDF from {url} ({len(pdf_bytes)} bytes)")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading PDF from {url}: Status {e.response.status_code}", exc_info=True)
            raise ScrapingError(f"Failed to download PDF from {url}: HTTP Status {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Network error downloading PDF from {url}: {e}", exc_info=True)
            raise ScrapingError(f"Network error downloading PDF from {url}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF from {url}: {e}", exc_info=True)
            raise ScrapingError(f"Unexpected download error for PDF {url}: {e}") from e

    # Ensure pdf_bytes is not None before proceeding (shouldn't happen if download is successful)
    if pdf_bytes is None:
         raise ScrapingError(f"PDF download resulted in None bytes for {url}")

    # 3. Optionally save the downloaded PDF
    full_file_path = None # Initialize outside the block
    if download_pdfs:
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            file_name = Path(urlparse(url).path).name
            if not file_name.lower().endswith('.pdf'): file_name += ".pdf"
            file_name = "".join(c for c in file_name if c.isalnum() or c in ('.', '-', '_')).rstrip()
            if not file_name: file_name = "downloaded_pdf.pdf"
            full_file_path = save_path / file_name # Assign the path here
            logger.info(f"Saving downloaded PDF to: {full_file_path}")
            with open(full_file_path, 'wb') as f: f.write(pdf_bytes)
        except OSError as e:
            logger.warning(f"Failed to save PDF ({full_file_path}), proceeding with extraction. Error: {e}")
            full_file_path = None # Reset path if PDF saving failed
        except Exception as e:
            logger.warning(f"Unexpected error saving PDF ({full_file_path}), proceeding. Error: {e}")
            full_file_path = None # Reset path if PDF saving failed

    # 4. Extract Markdown using MarkItDown
    try:
        logger.debug(f"Starting PDF parsing with MarkItDown for {url}")
        pdf_stream = io.BytesIO(pdf_bytes)
        loop = asyncio.get_running_loop()

        # MarkItDown.convert might be synchronous, run in executor
        # We pass the stream directly, as suggested by MarkItDown docs for streams
        result = await loop.run_in_executor(None, md_converter.convert, pdf_stream)

        pdf_stream.close()

        # Check if conversion was successful and extract text_content
        md_text: Optional[str] = None
        if result and hasattr(result, 'text_content'):
             md_text = result.text_content
        elif result: # If result exists but no text_content, log warning
             logger.warning(f"MarkItDown conversion for {url} succeeded but returned no text_content. Result: {result}")
        else: # If result itself is None/False
             logger.warning(f"MarkItDown conversion for {url} failed or returned None.")

        if not md_text: # Covers None or empty string
            logger.warning(f"MarkItDown extracted no text content from PDF: {url}")
            return ""

        logger.info(f"Successfully extracted Markdown using MarkItDown from PDF: {url} (Length: {len(md_text)})")

        # --- Save Extracted Markdown if PDF was saved --- #
        if download_pdfs and full_file_path and md_text: # Check if PDF path is valid and we have text
            try:
                md_file_path = full_file_path.with_suffix('.md')
                logger.info(f"Saving extracted PDF Markdown to: {md_file_path}")
                with open(md_file_path, 'w', encoding='utf-8') as f:
                    f.write(md_text)
            except Exception as e:
                logger.warning(f"Failed to save extracted PDF markdown to {md_file_path}: {e}")
                # Do not fail the overall process if markdown saving fails
        return md_text

    except Exception as e:
        logger.error(f"MarkItDown failed to process PDF stream from {url}: {e}", exc_info=True)
        raise ScrapingError(f"Failed to extract Markdown from PDF {url} using MarkItDown: {e}") from e
