"""Application-wide constants (no magic strings or numbers in business logic)."""

API_V1_PREFIX = "/v1"

# Document ingestion
DOCUMENT_LIST_LIMIT = 100
THREAD_LIST_LIMIT = 50
DEFAULT_UPLOAD_FILENAME = "upload.txt"
DEFAULT_TEXT_CONTENT_TYPE = "text/plain"
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DOCUMENT_STATUS_PROCESSING = "processing"
DOCUMENT_STATUS_INGESTED = "ingested"

# Async ingestion job lifecycle (ingestion_jobs table)
INGESTION_JOB_QUEUED = "queued"
INGESTION_JOB_PROCESSING = "processing"
INGESTION_JOB_COMPLETED = "completed"
INGESTION_JOB_FAILED = "failed"

# Agent routes (graph)
ROUTE_DIRECT = "direct"
ROUTE_SINGLE_HOP_RAG = "single_hop_rag"
ROUTE_MULTI_HOP = "multi_hop"

# Graph node names (logging / debug metadata)
GRAPH_NODE_ROUTE = "route"
GRAPH_NODE_RETRIEVE = "retrieve"
GRAPH_NODE_GRADE = "grade_context"
GRAPH_NODE_GENERATE = "generate"
GRAPH_NODE_VALIDATE = "validate_answer"

# LangGraph / agent messages
THREAD_TITLE_MAX_CHARS = 80
CITATION_SNIPPET_MAX_CHARS = 240
ABSTAIN_MESSAGE = (
    "I don't have enough information in the indexed documents to answer that."
)
DIRECT_GREETING_RESPONSE = (
    "Hello! I am ContextForge. Upload policy documents and ask questions "
    "about them — I will cite sources from your corpus."
)

# HTTP error messages (client-facing)
ERROR_DOCUMENT_NOT_FOUND = "Document not found"
ERROR_THREAD_NOT_FOUND = "Thread not found"
ERROR_INTERNAL_SERVER = "Internal server error"
ERROR_INVALID_UPLOAD = "Upload could not be processed"
ERROR_UPLOAD_TOO_LARGE = "Upload exceeds maximum allowed size"

# Chunking (recursive splitter)
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 120
DEFAULT_MARKDOWN_CONTENT_TYPE = "text/markdown"

# Rerank backends
RERANK_BACKEND_LEXICAL = "lexical"
RERANK_BACKEND_CROSS_ENCODER = "cross_encoder"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval store backends (see app/retrieval/factory.py)
RETRIEVAL_BACKEND_QDRANT = "qdrant"
RETRIEVAL_BACKEND_POSTGRES = "postgres"

# Generation context + citations (top reranked chunks after score threshold)
MAX_GENERATION_CONTEXTS = 3
# Additional chunks must be at least this fraction of the top rerank score (drops weak tail).
CITATION_SCORE_RELATIVE_MIN = 0.9

# Multi-turn chat
CONVERSATION_HISTORY_LIMIT = 10
CONVERSATION_RETRIEVAL_QUERY_MAX_CHARS = 512

# RRF default
RRF_RANK_CONSTANT = 60

# Lexical rerank blend weights
RERANK_BASE_SCORE_WEIGHT = 0.7
RERANK_LEXICAL_OVERLAP_WEIGHT = 0.3
LEXICAL_OVERLAP_EPSILON = 1e-9

# Server-sent events (POST /v1/query/stream)
SSE_EVENT_TOKEN = "token"
SSE_EVENT_STATUS = "status"
SSE_EVENT_DONE = "done"
SSE_EVENT_ERROR = "error"
SSE_STAGE_STARTED = "started"

# Pipeline sub-stages streamed live to the UI (see app/graph/progress.py).
# Node-complete events still use the GRAPH_NODE_* names above; these mark the
# *start* of a stage (phase="start") so the UI can highlight work in progress.
STAGE_REWRITE = "rewrite"
STAGE_ROUTE = "route"
STAGE_RETRIEVE_SEARCH = "retrieve.search"
STAGE_RETRIEVE_RERANK = "retrieve.rerank"
STAGE_GRADE = "grade"
STAGE_GRADE_JUDGE = "grade.judge"
STAGE_GENERATE_LLM = "generate.llm"
STAGE_VALIDATE = "validate"
STAGE_PHASE_START = "start"
STAGE_PHASE_END = "end"
SSE_STREAM_WORD_CHUNK_SIZE = 1
SSE_STREAM_CHUNK_DELAY_SECONDS = 0.04
SSE_FLUSH_COMMENT = ": flush\n\n"
