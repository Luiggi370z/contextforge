"""Application-wide constants (no magic strings or numbers in business logic)."""

API_V1_PREFIX = "/v1"

# Document ingestion
DOCUMENT_LIST_LIMIT = 100
THREAD_LIST_LIMIT = 50
DEFAULT_UPLOAD_FILENAME = "upload.txt"
DEFAULT_TEXT_CONTENT_TYPE = "text/plain"
DOCUMENT_STATUS_PROCESSING = "processing"
DOCUMENT_STATUS_INGESTED = "ingested"

# Agent routes (graph)
ROUTE_DIRECT = "direct"
ROUTE_SINGLE_HOP_RAG = "single_hop_rag"
ROUTE_MULTI_HOP = "multi_hop"
ROUTE_UNKNOWN = "unknown"

# Graph node names (logging / debug metadata)
GRAPH_NODE_ROUTE = "route"
GRAPH_NODE_RETRIEVE = "retrieve"
GRAPH_NODE_GRADE = "grade_context"
GRAPH_NODE_GENERATE = "generate"
GRAPH_NODE_VALIDATE = "validate_answer"

# HTTP error messages (client-facing)
ERROR_DOCUMENT_NOT_FOUND = "Document not found"
ERROR_THREAD_NOT_FOUND = "Thread not found"
ERROR_INTERNAL_SERVER = "Internal server error"
ERROR_INVALID_UPLOAD = "Upload could not be processed"

# RRF default
RRF_RANK_CONSTANT = 60
