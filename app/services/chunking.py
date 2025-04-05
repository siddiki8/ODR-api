from typing import List, Dict, Any
import re
import logging

# Import custom exceptions
from ..core.exceptions import ChunkingError

logger = logging.getLogger(__name__)

"""
Chonk
"""

# TODO: Implement or paste the Chunker class definition here.
# It should likely have a method like `chunk(text: str) -> List[str]`

class Chunker:
    """
    Chunker service that splits documents into smaller, semantically meaningful chunks.
    Supports recursive chunking for hierarchical document processing.
    """
    
    def __init__(self, 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100,
                 max_chunks: int = 1000):
        """
        Initialize the chunker with size parameters.
        
        Args:
            chunk_size: Target size for chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a valid chunk
            max_chunks: Maximum number of chunks to return (default: 1000, below rerank API limit of 1024)

        Raises:
            ValueError: If chunk sizes are invalid (e.g., negative, overlap too large).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size cannot be larger than chunk_size.")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunks = max_chunks
        logger.debug(f"Chunker initialized with size={chunk_size}, overlap={chunk_overlap}, "
                    f"min_size={min_chunk_size}, max_chunks={max_chunks}")
    
    def _strip_html(self, text: str) -> str:
        """
        Strip HTML tags from text before chunking.
        This is a simple implementation - more comprehensive HTML cleaning should be 
        done before the text reaches the chunker.
        """
        if not text:
            return ""
            
        # Log only the length, not the content
        input_length = len(text)
        logger.debug(f"Stripping HTML from text of length {input_length} chars")
        
        # Strip common HTML tags
        import re
        # Remove script, style tags and their content
        text = re.sub(r'<script.*?>.*?</script>', ' ', text, flags=re.DOTALL|re.IGNORECASE)
        text = re.sub(r'<style.*?>.*?</style>', ' ', text, flags=re.DOTALL|re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Handle HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        
        result = text.strip()
        output_length = len(result)
        logger.debug(f"HTML stripped: removed {input_length - output_length} chars ({(input_length - output_length) / max(1, input_length) * 100:.1f}%)")
        
        return result
    
    def chunk(self, text: str, recursive: bool = False, depth: int = 0) -> List[str]:
        """
        Split text into chunks with optional recursive chunking for large sections.
        
        Args:
            text: The text to chunk
            recursive: Whether to recursively chunk large sections
            depth: Current recursion depth (used internally)
            
        Returns:
            List of text chunks, limited to max_chunks
        """
        if not text: # Handle None or empty string
            logger.debug("Input text is empty or None, returning empty chunk list.")
            return []
            
        # First, ensure all HTML is stripped
        original_length = len(text)
        text = self._strip_html(text)
        stripped_length = len(text)
        
        if original_length > stripped_length:
            logger.debug(f"Stripped HTML: removed {original_length - stripped_length} chars ({(original_length - stripped_length) / max(1, original_length) * 100:.1f}%)")
        
        # If text is smaller than min size, return as is (if not empty)
        if len(text) < self.min_chunk_size:
            logger.debug(f"Input text length ({len(text)}) is below min_chunk_size ({self.min_chunk_size}). Returning as single chunk.")
            return [text]
        
        # Simple case: text fits in a single chunk
        if len(text) <= self.chunk_size:
            logger.debug(f"Input text length ({len(text)}) fits within chunk_size ({self.chunk_size}). Returning as single chunk.")
            return [text]
        
        try:
            # Try splitting by natural boundaries in decreasing order of preference
            logger.debug("Attempting to split by headings...")
            chunks = self._split_by_headings(text)
            
            if not chunks or (len(chunks) == 1 and chunks[0] == text):
                logger.debug("Heading split ineffective, attempting to split by paragraphs...")
                chunks = self._split_by_paragraphs(text)
            
            if not chunks or (len(chunks) == 1 and chunks[0] == text):
                logger.debug("Paragraph split ineffective, attempting to split by sentences...")
                chunks = self._split_by_sentences(text)
            
            if not chunks or (len(chunks) == 1 and chunks[0] == text):
                logger.debug("Sentence split ineffective, falling back to character split...")
                chunks = self._split_by_chars(text)
            
            # If recursive and we have large chunks, process them further
            if recursive and depth < 3: # Slightly increased max depth
                logger.debug(f"Recursively chunking {len(chunks)} initial chunks at depth {depth}.")
                result_chunks = []
                for chunk in chunks:
                    if len(chunk) > self.chunk_size:
                        # Recursively chunk large sections
                        result_chunks.extend(self.chunk(chunk, recursive=True, depth=depth+1))
                    elif len(chunk) >= self.min_chunk_size:
                        result_chunks.append(chunk)
                    else:
                        logger.debug(f"Skipping recursive chunk result below min_chunk_size ({len(chunk)} chars).")
                logger.debug(f"Finished recursive chunking at depth {depth}. Total chunks: {len(result_chunks)}.")
                chunks = result_chunks
            else:
                # Filter out chunks below min size even if not recursive
                chunks = [c for c in chunks if len(c) >= self.min_chunk_size]
                if len(chunks) < len(chunks):
                    logger.debug(f"Filtered chunks below min_chunk_size after splitting.")

            # Enforce max_chunks limit if needed
            if len(chunks) > self.max_chunks:
                logger.warning(f"Chunking produced {len(chunks)} chunks, exceeding max_chunks limit of {self.max_chunks}. "
                              f"Truncating to {self.max_chunks} chunks.")
                chunks = chunks[:self.max_chunks]
                
            logger.debug(f"Splitting finished. Total chunks: {len(chunks)}.")
            return chunks

        except re.error as e:
            logger.error(f"Regex error during chunking: {e}", exc_info=True)
            raise ChunkingError(f"Regex error occurred during text splitting: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during chunking process: {e}", exc_info=True)
            raise ChunkingError(f"An unexpected error occurred during chunking: {e}") from e
    
    def _split_by_headings(self, text: str) -> List[str]:
        """Split text at heading boundaries (markdown or HTML headings)."""
        # Regex to find headings, ensuring they are preceded by a newline or start of string
        heading_pattern = r'(?:^|\r?\n)(#{1,6}\s+.+|<h[1-6]>.*?</h[1-6]>)'
        try:
            matches = list(re.finditer(heading_pattern, text, re.IGNORECASE))
        except re.error as e:
            logger.error(f"Regex error finding headings: {e}")
            raise # Re-raise to be caught by the main chunk method

        if len(matches) <= 1:
            logger.debug("Found <= 1 heading match, cannot split by headings.")
            return [] # Return empty list, not list containing original text

        chunks = []
        start_index = 0
        for match in matches:
            # Chunk before the heading
            chunk = text[start_index:match.start()].strip()
            if chunk and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            # Update start index to the beginning of the heading for the next potential chunk
            start_index = match.start()
        
        # Add the final chunk (from the last heading to the end)
        final_chunk = text[start_index:].strip()
        if final_chunk and len(final_chunk) >= self.min_chunk_size:
            chunks.append(final_chunk)

        logger.debug(f"Split by headings resulted in {len(chunks)} potential chunks.")
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text at paragraph boundaries."""
        try:
            paragraphs = re.split(r'(\r?\n){2,}', text)
        except re.error as e:
            logger.error(f"Regex error splitting paragraphs: {e}")
            raise

        chunks = []
        current_chunk = ""
        
        for p in paragraphs:
            p_stripped = p.strip()
            if not p_stripped:
                continue
                
            if len(current_chunk) + len(p_stripped) + 2 > self.chunk_size: # +2 for newline chars
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = p_stripped # Start new chunk
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + p_stripped
                else:
                    current_chunk = p_stripped
        
        # Add the last chunk if it's valid
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)

        logger.debug(f"Split by paragraphs resulted in {len(chunks)} potential chunks.")
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text at sentence boundaries."""
        # Improved sentence splitting using lookbehind for punctuation followed by space/newline
        try:
            sentences = re.split(r'(?<=[.?!])(\s+|\r?\n)', text)
        except re.error as e:
            logger.error(f"Regex error splitting sentences: {e}")
            raise

        chunks = []
        current_chunk = ""
        current_sentence_parts = []
        
        for part in sentences:
            if not part or part.isspace(): # Skip empty/whitespace parts from split
                continue
            current_sentence_parts.append(part)
            sentence = "".join(current_sentence_parts).strip()

            # If sentence ends with punctuation, process it
            if sentence.endswith(tuple('.?!')):
                if current_chunk and len(current_chunk) + len(sentence) + 1 > self.chunk_size: # +1 for space
                    if len(current_chunk) >= self.min_chunk_size:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                current_sentence_parts = [] # Reset for next sentence
            # Else, keep accumulating parts (e.g., if split occurred mid-sentence)

        # Add remaining parts as the last sentence/chunk
        last_sentence = "".join(current_sentence_parts).strip()
        if last_sentence:
            if current_chunk and len(current_chunk) + len(last_sentence) + 1 <= self.chunk_size:
                current_chunk += " " + last_sentence
            elif len(last_sentence) >= self.min_chunk_size:
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = last_sentence # Assign last sentence as new chunk
            # else: last part is too short, gets merged or dropped depending on current_chunk state

        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)

        logger.debug(f"Split by sentences resulted in {len(chunks)} potential chunks.")
        return chunks
    
    def _split_by_chars(self, text: str) -> List[str]:
        """Fall back to simple character-based chunking with overlap."""
        chunks = []
        text_len = len(text)
        start_pos = 0
        while start_pos < text_len:
            end_pos = min(start_pos + self.chunk_size, text_len)

            # Try to backtrack to the nearest space if not at the end
            if end_pos < text_len:
                last_space = text.rfind(' ', start_pos, end_pos)
                # Only backtrack if it doesn't make the chunk too small and space exists
                if last_space > start_pos and end_pos - last_space < self.chunk_size - self.min_chunk_size:
                    end_pos = last_space + 1 # Include the space for potential sentence start
            
            chunk = text[start_pos:end_pos].strip()
            if chunk and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position for the next chunk, considering overlap
            next_start = start_pos + self.chunk_size - self.chunk_overlap
            # Ensure progress is made
            if next_start <= start_pos:
                next_start = start_pos + 1 
            start_pos = next_start
            # Break if start_pos exceeds length (handles edge cases)
            if start_pos >= text_len:
                break

        logger.debug(f"Split by characters resulted in {len(chunks)} chunks.")
        return chunks
    
    def chunk_and_label(self, documents: List[Dict[str, Any]], sequential: bool = False) -> List[Dict[str, Any]]:
        """
        Process a list of documents, chunking each and preserving metadata.
        
        Args:
            documents: List of document dictionaries, expected to have a 'content' field.
            sequential: If True, process documents sequentially to avoid rate limits
            
        Returns:
            List of chunk dictionaries with original metadata plus chunk_id.
            Returns an empty list if input is empty or documents lack 'content'.
            The total number of chunks will not exceed max_chunks.

        Raises:
            ChunkingError: If chunking fails for any document.
            ValueError: If input `documents` is not a list.
        """
        if not isinstance(documents, list):
            raise ValueError("Input documents must be a list of dictionaries.")
        if not documents:
            return []

        chunked_docs = []
        total_chunks = 0
        max_chunks_reached = False
        
        for doc_idx, doc in enumerate(documents):
            if max_chunks_reached:
                logger.warning(f"Max chunks limit ({self.max_chunks}) reached. Skipping processing of remaining documents.")
                break
                
            if not isinstance(doc, dict):
                logger.warning(f"Skipping item at index {doc_idx} in documents list because it is not a dictionary.")
                continue
            
            content = doc.get('content')
            if content is None:
                logger.warning(f"Skipping document index {doc_idx} because it lacks a 'content' field.")
                continue
            if not isinstance(content, str):
                logger.warning(f"Skipping document index {doc_idx} because 'content' field is not a string (type: {type(content).__name__}).")
                continue

            doc_id_info = doc.get('id', f'doc{doc_idx}') # Use ID if available, else index
            try:
                chunks = self.chunk(content, recursive=True)
                
                # Check if adding these chunks would exceed max_chunks
                remaining_capacity = self.max_chunks - total_chunks
                if len(chunks) > remaining_capacity:
                    logger.warning(f"Document '{doc_id_info}' would produce {len(chunks)} chunks, but only {remaining_capacity} "
                                 f"more chunks allowed before reaching max_chunks limit of {self.max_chunks}. Truncating.")
                    chunks = chunks[:remaining_capacity]
                    max_chunks_reached = True
                
                logger.debug(f"Chunked document '{doc_id_info}' into {len(chunks)} chunks.")
                total_chunks += len(chunks)

                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_doc = doc.copy() # Shallow copy metadata
                    chunk_doc['content'] = chunk_text
                    chunk_doc['chunk_id'] = f"{doc_idx}-{chunk_idx}" # Simple chunk ID
                    chunk_doc['is_chunk'] = True
                    chunked_docs.append(chunk_doc)
                    
                if max_chunks_reached:
                    logger.warning(f"Max chunks limit of {self.max_chunks} reached after processing document '{doc_id_info}'.")
                    break
                    
            except ChunkingError as e:
                logger.error(f"Failed to chunk document index {doc_idx} ('{doc_id_info}'): {e}")
                # Propagate the error - failure to chunk one doc should fail the whole batch?
                # Or collect errors and return partial results? Raising for now.
                raise ChunkingError(f"Failed to process document {doc_idx}: {e}") from e
            except Exception as e:
                # Catch unexpected errors during labeling/copying
                logger.error(f"Unexpected error processing document index {doc_idx} ('{doc_id_info}') after chunking: {e}", exc_info=True)
                raise ChunkingError(f"Unexpected error processing document {doc_idx}: {e}") from e

        logger.info(f"Finished chunking and labeling {len(documents)} documents into {total_chunks} total chunks.")
        return chunked_docs 