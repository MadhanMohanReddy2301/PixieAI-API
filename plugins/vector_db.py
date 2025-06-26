import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from semantic_kernel.functions import kernel_function

# Constants: Hardcoded paths for FAISS index and metadata
INDEX_PATH = "plugins/vectorstore/faiss.index"
METADATA_PATH = "plugins/vectorstore/metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

class FaissSemanticSearchSkill:
    """
    Semantic Kernel skill for retrieving document chunks from a FAISS vector store.
    Paths are hardcoded; no init parameters required.
    """
    def __init__(self, top_k: int = DEFAULT_TOP_K):

        # Load FAISS index and metadata
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        # Initialize embedding model
        self.model = SentenceTransformer(EMBED_MODEL)
        self.top_k = top_k

    @kernel_function(
        name="retrieve_related_docs",
        description="Retrieve the top-k most relevant document chunks based on the input query."
    )
    def retrieve_related_docs(self, query: str) -> str:
        """
        Args:
            query: The user’s search query.
        Returns:
            A concatenated string of the top-k document snippets with source labels.
        """
        if not query or not query.strip():
            return "⚠️ Please provide a valid query."

        # Embed and normalize
        emb = self.model.encode([query])
        faiss.normalize_L2(emb)

        # FAISS search
        _, indices = self.index.search(emb, self.top_k)

        # Format snippets
        snippets = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                source = meta.get("source", "unknown")
                chunk = meta.get("chunk", "").replace("\n", " ")
                snippet = chunk[:200].strip() + ("..." if len(chunk) > 200 else "")
                snippets.append(f"📄 {source}: {snippet}")

        return "\n\n".join(snippets) or "No relevant documents found."