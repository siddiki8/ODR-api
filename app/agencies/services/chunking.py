"""
Service for chunking text documents using LangChain text splitters.

Replaces the previous custom Chunker class with a function leveraging
MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter for 
better semantic chunking, especially for Markdown content.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document type

# Pydantic imports for Chunk schema
from pydantic import BaseModel, Field, HttpUrl, ConfigDict

# Import custom exceptions using absolute path
from app.core.exceptions import ChunkingError

logger = logging.getLogger(__name__)

def chunk_document( 
    doc_content: str, 
    metadata: Dict[str, Any], # Metadata to attach to chunks
    chunk_size: int = 2048, # Target chunk size (chars)
    chunk_overlap: int = 100, # Overlap between chunks (chars)
    min_chunk_size: int = 256, # Minimum size for a chunk to be kept
) -> List[Document]:
    """
    Chunks a single document string using LangChain splitters.

    First attempts to split based on Markdown headers, then uses recursive character 
    splitting on any resulting sections that are still larger than chunk_size.

    Args:
        doc_content: The string content of the document.
        metadata: Original metadata to be attached to each resulting chunk Document.
        chunk_size: Target maximum size for each chunk (in characters).
        chunk_overlap: Character overlap between consecutive chunks.
        min_chunk_size: Minimum character length for a chunk to be kept.

    Returns:
        A list of LangChain Document objects representing the chunks.
    
    Raises:
        ChunkingError: Wraps exceptions raised during the splitting process.
    """
    if not doc_content or not isinstance(doc_content, str):
        logger.warning("Received empty or non-string document content, returning no chunks.")
        return []

    final_chunks = []
    try:
        # Define headers to split on (adjust levels as needed)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False # Keep header text in the chunk content
        )
        
        # Split by headers
        md_header_splits = markdown_splitter.split_text(doc_content)
        logger.debug(f"Markdown split created {len(md_header_splits)} initial sections.")

        # Prepare recursive splitter for sections that are too large
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""] # Common separators
        )

        # Process each section from the header split
        for section_doc in md_header_splits:
            section_text = section_doc.page_content
            # Combine original metadata with any metadata from the header split
            combined_metadata = metadata.copy()
            combined_metadata.update(section_doc.metadata)

            if len(section_text) <= chunk_size and len(section_text) >= min_chunk_size:
                # Section is within size limits, add it directly
                final_chunks.append(Document(page_content=section_text, metadata=combined_metadata))
            elif len(section_text) > chunk_size:
                # Section is too large, split it further recursively
                logger.debug(f"Recursively splitting section (length {len(section_text)} > {chunk_size}) starting with: '{section_text[:80]}...'")
                recursive_splits = recursive_splitter.split_text(section_text)
                for split_content in recursive_splits:
                    if len(split_content) >= min_chunk_size:
                        # Create a Document for each valid recursive split, retaining combined metadata
                        final_chunks.append(Document(page_content=split_content, metadata=combined_metadata.copy()))
                    else:
                        logger.debug(f"Discarding recursive split result below min_chunk_size ({len(split_content)} chars).")
            else: # len(section_text) < min_chunk_size
                 logger.debug(f"Discarding header split section below min_chunk_size ({len(section_text)} chars).")
            
        logger.debug(f"Finished chunking document. Total valid chunks: {len(final_chunks)}.")
        return final_chunks

    except Exception as e:
        logger.error(f"Error during LangChain chunking process: {e}", exc_info=True)
        raise ChunkingError(f"An unexpected error occurred during chunking: {e}") from e

def chunk_and_label(
    documents: List[Dict[str, Any]],
    chunk_size: int = 2048,
    chunk_overlap: int = 100,
    min_chunk_size: int = 256,
    max_chunks: Optional[int] = 1000,
) -> List[Dict[str, Any]]:
    """
    Processes a list of document dictionaries, chunks each using chunk_document, 
    and returns a list of chunk dictionaries suitable for downstream processing.

    Args:
        documents: List of document dictionaries. Each must contain 'content' (str) 
                   and may contain other metadata (e.g., 'title', 'link').
        chunk_size: Target maximum size for each chunk (passed to chunk_document).
        chunk_overlap: Character overlap between chunks (passed to chunk_document).
        min_chunk_size: Minimum character length for chunks (passed to chunk_document).
        max_chunks: Optional maximum total number of chunks to return across all documents.

    Returns:
        List of chunk dictionaries. Each dictionary contains the original metadata 
        plus 'content' (the chunk text), 'chunk_id', and 'is_chunk'.
        Returns an empty list if input is empty or contains no processable documents.

    Raises:
        ValueError: If input `documents` is not a list.
        ChunkingError: Propagated from chunk_document if critical errors occur.
    """
    if not isinstance(documents, list):
        raise ValueError("Input documents must be a list of dictionaries.")
    if not documents:
        logger.info("Received empty documents list, returning empty results.")
        return []

    all_chunked_dicts = []
    total_chunks_generated = 0

    for doc_idx, doc_dict in enumerate(documents):
        
        # Check max_chunks limit before processing the next document
        if max_chunks is not None and total_chunks_generated >= max_chunks:
            logger.warning(f"Reached max_chunks limit ({max_chunks}). Skipping remaining {len(documents) - doc_idx} documents.")
            break
             
        if not isinstance(doc_dict, dict):
            logger.warning(f"Skipping item at index {doc_idx} in documents list: not a dictionary.")
            continue
            
        content = doc_dict.get('content')
        if not content or not isinstance(content, str):
            logger.warning(f"Skipping document index {doc_idx}: missing, empty, or invalid 'content' field.")
            continue

        # Prepare metadata for chunking, exclude original full content
        metadata = {k: v for k, v in doc_dict.items() if k != 'content'}
        # Use ID or index for logging/tracking
        doc_id_info = metadata.get('id', f'doc{doc_idx}') 

        try:
            logger.info(f"Chunking document index {doc_idx} ('{doc_id_info}')...")
            # Call the refactored chunking function which returns List[Document]
            langchain_chunks: List[Document] = chunk_document(
                doc_content=content, 
                metadata=metadata, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size
            )
            
            doc_chunks_added = 0
            for chunk_idx, lc_chunk in enumerate(langchain_chunks):
                # Check max_chunks limit before adding each chunk
                if max_chunks is not None and total_chunks_generated >= max_chunks:
                     logger.warning(f"Reached max_chunks limit ({max_chunks}) while processing chunks for document '{doc_id_info}'. Truncating further chunks from this document.")
                     break # Stop adding chunks from this doc
                     
                # Convert LangChain Document back to dictionary format expected by agent
                chunk_dict = lc_chunk.metadata.copy() # Start with metadata
                chunk_dict['content'] = lc_chunk.page_content
                chunk_dict['chunk_id'] = f"{doc_idx}-{chunk_idx}" # Assign unique chunk ID
                chunk_dict['is_chunk'] = True # Label as a chunk
                all_chunked_dicts.append(chunk_dict)
                total_chunks_generated += 1
                doc_chunks_added += 1
                
            logger.info(f"Document '{doc_id_info}' processed into {doc_chunks_added} valid chunks.")

        except ChunkingError as e:
            # Log specific chunking errors but allow processing to continue
            logger.error(f"Failed to chunk document index {doc_idx} ('{doc_id_info}'): {e}", exc_info=False)
        except Exception as e:
            # Catch unexpected errors during the processing of a single document
            logger.error(f"Unexpected error processing document index {doc_idx} ('{doc_id_info}'): {e}", exc_info=True)
            
    logger.info(f"Finished processing {len(documents)} documents. Generated {total_chunks_generated} total chunks.")
    # Apply final max_chunks limit just in case concurrent processing exceeds slightly (though less likely now)
    if max_chunks is not None and len(all_chunked_dicts) > max_chunks:
        logger.warning(f"Final chunk list ({len(all_chunked_dicts)}) exceeds max_chunks ({max_chunks}). Truncating final list.")
        all_chunked_dicts = all_chunked_dicts[:max_chunks]
        
    return all_chunked_dicts

# --- Removed Old Chunker Class --- 

# --- Schemas ---

class Chunk(BaseModel):
    """Represents a single chunk of relevant text extracted from a source document."""
    model_config = ConfigDict(extra='ignore')
    content: str = Field(..., description="The text content of the chunk.", min_length=1)
    link: HttpUrl = Field(..., description="The URL of the original source document.")
    title: str = Field(..., description="The title of the original source document.")
    relevance_score: Optional[float] = Field(None, description="Score assigned by the reranker indicating relevance to the query.")
    rank: Optional[int] = Field(None, description="Sequential rank assigned during processing (e.g., for citation).") 