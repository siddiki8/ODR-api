import asyncio
import logging
import os
from pathlib import Path
import httpx
import io
from urllib.parse import urlparse
from typing import Optional
import fitz # PyMuPDF

# Use MarkItDown instead of pymupdf4llm
from markitdown import MarkItDown
# import pymupdf4llm # Removed

# Import custom exceptions relative to the app root
from ...core.exceptions import ScrapingError, ConfigurationError

logger = logging.getLogger(__name__)

# Initialize MarkItDown once (can be reused)
# Pass enable_plugins directly
md_converter = MarkItDown(enable_plugins=False)

# Default timeout for HTTP requests (seconds)
DEFAULT_TIMEOUT = 30

async def handle_local_pdf_file(
    file_path: str,
    max_size_bytes: Optional[int] = None
) -> str:
    """
    Reads a local PDF file, checks its size, and extracts Markdown content using MarkItDown.

    Args:
        file_path: The path to the local PDF file.
        max_size_bytes: Optional maximum size in bytes. If exceeded, extraction is skipped.

    Returns:
        The extracted Markdown content of the PDF, or an empty string if skipped or failed.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        ScrapingError: If parsing the PDF fails.
    """
    logger.info(f"Attempting to handle local PDF file with MarkItDown: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Local PDF file not found: {file_path}")

    # 1. Check PDF Size
    try:
        file_size = os.path.getsize(file_path)
        if max_size_bytes is not None and file_size > max_size_bytes:
            logger.warning(
                f"Local PDF size ({file_size / 1024 / 1024:.2f} MB) exceeds limit "
                f"({max_size_bytes / 1024 / 1024:.2f} MB). Skipping: {file_path}"
            )
            return "" # Skipped due to size
        logger.debug(f"Local PDF size ({file_size / 1024 / 1024:.2f} MB) is within limit for {file_path}")
    except OSError as e:
         logger.error(f"Could not get size of local PDF file {file_path}: {e}")
         raise ScrapingError(f"Failed to check size of local PDF {file_path}: {e}") from e

    # 2. Read PDF bytes from local file
    try:
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        if not pdf_bytes:
             logger.warning(f"Read empty file from local path: {file_path}")
             return "" # Treat empty file as no content
        logger.debug(f"Successfully read local PDF: {file_path} ({len(pdf_bytes)} bytes)")
    except OSError as e:
        logger.error(f"Failed to read local PDF file {file_path}: {e}")
        raise ScrapingError(f"Failed to read local PDF {file_path}: {e}") from e
    except Exception as e:
         logger.error(f"Unexpected error reading local PDF file {file_path}: {e}", exc_info=True)
         raise ScrapingError(f"Unexpected error reading local PDF {file_path}: {e}") from e

    # 3. Extract Markdown using MarkItDown
    try:
        logger.debug(f"Starting PDF parsing with MarkItDown for local file: {file_path}")
        pdf_stream = io.BytesIO(pdf_bytes)
        loop = asyncio.get_running_loop()

        # MarkItDown.convert might be synchronous, run in executor
        result = await loop.run_in_executor(None, md_converter.convert, pdf_stream)
        pdf_stream.close()

        md_text: Optional[str] = None
        if result and hasattr(result, 'text_content'):
             md_text = result.text_content
        elif result: 
             logger.warning(f"MarkItDown conversion for {file_path} succeeded but returned no text_content. Result: {result}")
        else:
             logger.warning(f"MarkItDown conversion for {file_path} failed or returned None.")

        if not md_text:
            logger.warning(f"MarkItDown extracted no text content from local PDF: {file_path}")
            return ""

        logger.info(f"Successfully extracted Markdown using MarkItDown from local PDF: {file_path} (Length: {len(md_text)})")
        return md_text

    except Exception as e:
        logger.error(f"MarkItDown failed to process local PDF stream from {file_path}: {e}", exc_info=True)
        raise ScrapingError(f"Failed to extract Markdown from local PDF {file_path} using MarkItDown: {e}") from e

async def handle_pdf_url(
    url: str,
    download_pdfs: bool,
    save_dir: str,
    max_size_bytes: int,
    timeout: int = DEFAULT_TIMEOUT
) -> Optional[str]:
    """Handles fetching and extracting text from a PDF URL.

    Args:
        url: The URL of the PDF file.
        download_pdfs: Whether to save the downloaded PDF.
        save_dir: Directory to save downloaded PDFs.
        max_size_bytes: Maximum allowed size for the PDF download.
        timeout: Request timeout in seconds.

    Returns:
        Extracted text content as a string, or None if fetching/processing fails.

    Raises:
        ScrapingError: For issues like file parsing errors after successful download.
    """
    extracted_text: Optional[str] = None

    # Ensure save directory exists if downloading
    if download_pdfs and not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError as e:
            logger.error(f"Failed to create PDF save directory '{save_dir}': {e}")
            # Return None as we cannot proceed with saving if requested
            return None

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            logger.debug(f"Requesting PDF HEAD from: {url}")
            try:
                head_response = await client.head(url)
                head_response.raise_for_status() # Check HEAD response first
            except httpx.HTTPStatusError as e:
                 logger.warning(f"Failed to fetch PDF HEAD from {url}: Status {e.response.status_code}")
                 return None # Cannot proceed if HEAD fails
            except httpx.RequestError as e:
                 logger.warning(f"Network error fetching PDF HEAD from {url}: {e}")
                 return None

            # Check Content-Type
            content_type = head_response.headers.get("content-type", "").lower()
            if "application/pdf" not in content_type:
                logger.warning(f"Expected PDF content type, but got '{content_type}' for URL: {url}")
                # Don't treat as hard error, but maybe skip? For now, proceed but log.
                # return None # Optional: uncomment to strictly enforce content-type

            # Check Content-Length
            content_length = head_response.headers.get("content-length")
            if content_length:
                try:
                    pdf_size = int(content_length)
                    if pdf_size > max_size_bytes:
                        logger.warning(f"PDF size ({pdf_size} bytes) exceeds maximum ({max_size_bytes} bytes) for URL: {url}")
                        return None # Skip large file
                except ValueError:
                    logger.warning(f"Could not parse Content-Length header: {content_length} for {url}")
                    # Proceed cautiously if size unknown

            # If checks pass, download the full PDF
            logger.info(f"Downloading PDF from: {url}")
            try:
                response = await client.get(url)
                response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)
            except httpx.HTTPStatusError as e:
                 # Log warning and return None for HTTP errors (4xx/5xx)
                 logger.warning(f"Failed to download PDF from {url}: Status {e.response.status_code}")
                 return None
            except httpx.RequestError as e:
                 # Log warning and return None for network errors
                 logger.warning(f"Network error downloading PDF from {url}: {e}")
                 return None

            # Process the downloaded content
            pdf_content = response.content

            # Save the PDF if requested
            if download_pdfs:
                filename = url.split('/')[-1]
                if not filename.lower().endswith('.pdf'):
                    filename += ".pdf" # Ensure .pdf extension
                # Basic sanitization (replace potential path traversal chars)
                filename = filename.replace("../", "_").replace("/", "_")
                save_path = os.path.join(save_dir, filename)
                try:
                    with open(save_path, 'wb') as f:
                        f.write(pdf_content)
                    logger.info(f"PDF saved to: {save_path}")
                except IOError as e:
                    logger.error(f"Failed to save PDF to {save_path}: {e}")
                    # Continue with extraction even if saving fails, but log error

            # Extract text using PyMuPDF
            logger.info(f"Extracting text from PDF: {url}")
            try:
                pdf_document = fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf")
                # Optimize text extraction
                all_text = []
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    # Use extract_text with flags for better layout preservation if needed
                    # or simpler text extraction:
                    all_text.append(page.get_text("text", sort=True))
                    # Release page resources early if memory is a concern
                    # fitz.TOOLS.store_shrink(100) # Optional: Aggressive memory management

                extracted_text = "\n".join(all_text).strip()
                # Clean up common extraction artifacts
                extracted_text = extracted_text.replace("\t", " ") # Replace tabs with spaces
                extracted_text = '\n'.join(line.strip() for line in extracted_text.splitlines() if line.strip())

                pdf_document.close()
                logger.info(f"Successfully extracted ~{len(extracted_text)} chars from PDF: {url}")
            except Exception as e:
                logger.error(f"PyMuPDF failed to process PDF from {url}: {e}", exc_info=True)
                # Raise ScrapingError here, as this is a file processing error, not access error
                raise ScrapingError(f"Failed to parse PDF content from {url}: {e}") from e

    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"Unexpected error handling PDF URL {url}: {e}", exc_info=True)
        # It's safer to return None than raise a generic Exception
        # Or potentially re-raise as a ScrapingError if appropriate
        return None

    return extracted_text

async def handle_pdf_file(
    file_path: str,
    max_size_bytes: int
) -> str:
    """
    Reads a local PDF file, checks its size, and extracts Markdown content using MarkItDown.

    Args:
        file_path: The path to the local PDF file.
        max_size_bytes: Optional maximum size in bytes. If exceeded, extraction is skipped.

    Returns:
        The extracted Markdown content of the PDF, or an empty string if skipped or failed.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        ScrapingError: If parsing the PDF fails.
    """
    logger.info(f"Attempting to handle local PDF file with MarkItDown: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Local PDF file not found: {file_path}")

    # 1. Check PDF Size
    try:
        file_size = os.path.getsize(file_path)
        if max_size_bytes is not None and file_size > max_size_bytes:
            logger.warning(
                f"Local PDF size ({file_size / 1024 / 1024:.2f} MB) exceeds limit "
                f"({max_size_bytes / 1024 / 1024:.2f} MB). Skipping: {file_path}"
            )
            return "" # Skipped due to size
        logger.debug(f"Local PDF size ({file_size / 1024 / 1024:.2f} MB) is within limit for {file_path}")
    except OSError as e:
         logger.error(f"Could not get size of local PDF file {file_path}: {e}")
         raise ScrapingError(f"Failed to check size of local PDF {file_path}: {e}") from e

    # 2. Read PDF bytes from local file
    try:
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        if not pdf_bytes:
             logger.warning(f"Read empty file from local path: {file_path}")
             return "" # Treat empty file as no content
        logger.debug(f"Successfully read local PDF: {file_path} ({len(pdf_bytes)} bytes)")
    except OSError as e:
        logger.error(f"Failed to read local PDF file {file_path}: {e}")
        raise ScrapingError(f"Failed to read local PDF {file_path}: {e}") from e
    except Exception as e:
         logger.error(f"Unexpected error reading local PDF file {file_path}: {e}", exc_info=True)
         raise ScrapingError(f"Unexpected error reading local PDF {file_path}: {e}") from e

    # 3. Extract Markdown using MarkItDown
    try:
        logger.debug(f"Starting PDF parsing with MarkItDown for local file: {file_path}")
        pdf_stream = io.BytesIO(pdf_bytes)
        loop = asyncio.get_running_loop()

        # MarkItDown.convert might be synchronous, run in executor
        result = await loop.run_in_executor(None, md_converter.convert, pdf_stream)
        pdf_stream.close()

        md_text: Optional[str] = None
        if result and hasattr(result, 'text_content'):
             md_text = result.text_content
        elif result: 
             logger.warning(f"MarkItDown conversion for {file_path} succeeded but returned no text_content. Result: {result}")
        else:
             logger.warning(f"MarkItDown conversion for {file_path} failed or returned None.")

        if not md_text:
            logger.warning(f"MarkItDown extracted no text content from local PDF: {file_path}")
            return ""

        logger.info(f"Successfully extracted Markdown using MarkItDown from local PDF: {file_path} (Length: {len(md_text)})")
        return md_text

    except Exception as e:
        logger.error(f"MarkItDown failed to process local PDF stream from {file_path}: {e}", exc_info=True)
        raise ScrapingError(f"Failed to extract Markdown from local PDF {file_path} using MarkItDown: {e}") from e
