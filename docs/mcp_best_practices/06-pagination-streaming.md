# 6. Pagination, Streaming & Idempotency

> Handle large datasets and ensure safe retries with proper cursor and idempotency patterns.

## Overview

MCP tools often need to return large datasets or support streaming operations. This section covers pagination, streaming, and idempotency patterns that enable safe, efficient data transfer.

## Requirements

### MUST

- **Use cursor-based pagination** for list operations
- **Document idempotency guarantees** for each tool
- **Indicate partial results** when responses are truncated
- **Support safe replay** for idempotent operations

### SHOULD

- **Use opaque cursors** that don't expose implementation details
- **Include `has_more` indicator** in paginated responses
- **Set reasonable page size defaults** with configurable limits
- **Provide idempotency keys** for non-idempotent operations

### MAY

- **Support streaming responses** for real-time data
- **Include total counts** when efficiently available
- **Allow cursor expiration** with clear documentation

## Cursor-Based Pagination

### Why Cursors Over Offsets

| Approach | Pros | Cons |
|----------|------|------|
| **Offset** (`?page=3`) | Simple | Inconsistent with concurrent writes, slow for large offsets |
| **Cursor** (`?cursor=abc`) | Consistent, performant | More complex to implement |

### Cursor Design

```python
import base64
import json
from typing import Optional, Tuple, List, Any

def encode_cursor(data: dict) -> str:
    """Encode cursor data as opaque token."""
    return base64.urlsafe_b64encode(
        json.dumps(data).encode()
    ).decode()

def decode_cursor(cursor: str) -> dict:
    """Decode cursor token to data."""
    try:
        return json.loads(
            base64.urlsafe_b64decode(cursor.encode())
        )
    except Exception:
        raise ValueError("Invalid cursor")

# Example cursor containing last seen ID and timestamp
cursor_data = {
    "last_id": "item_123",
    "last_ts": 1732600000,
    "version": 1  # For cursor format versioning
}
cursor = encode_cursor(cursor_data)  # "eyJsYXN0X2lkIjogIml0ZW1fMTIzIi4uLn0="
```

### Pagination Implementation

```python
from dataclasses import asdict
from typing import Optional, List

DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000

@mcp.tool()
def list_items(
    cursor: Optional[str] = None,
    limit: int = DEFAULT_PAGE_SIZE
) -> dict:
    """List items with cursor-based pagination.

    Args:
        cursor: Pagination cursor from previous response
        limit: Number of items per page (max 1000)
    """
    # Validate limit
    limit = min(max(1, limit), MAX_PAGE_SIZE)

    # Decode cursor if provided
    start_after = None
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
            start_after = cursor_data["last_id"]
        except ValueError:
            return asdict(error_response("Invalid pagination cursor"))

    # Fetch items (request one extra to detect has_more)
    items = db.list_items(
        start_after=start_after,
        limit=limit + 1
    )

    # Determine if more items exist
    has_more = len(items) > limit
    if has_more:
        items = items[:limit]  # Remove extra item

    # Generate next cursor
    next_cursor = None
    if has_more and items:
        next_cursor = encode_cursor({
            "last_id": items[-1]["id"],
            "version": 1
        })

    return asdict(success_response(
        data={"items": items},
        pagination={
            "cursor": next_cursor,
            "has_more": has_more,
            "page_size": limit
        }
    ))
```

### Pagination Response Format

```json
{
    "success": true,
    "data": {
        "items": [
            {"id": "item_101", "name": "..."},
            {"id": "item_102", "name": "..."}
        ]
    },
    "meta": {
        "version": "response-v2",
        "pagination": {
            "cursor": "eyJsYXN0X2lkIjogIml0ZW1fMTAyIn0=",
            "has_more": true,
            "page_size": 100,
            "total_count": 5432
        }
    }
}
```

## Streaming Responses

### When to Stream

- Real-time event feeds
- Large file transfers
- Long-running operations with progress updates

### Streaming Pattern

```python
from typing import AsyncIterator
import json

async def stream_events(
    since_id: Optional[str] = None
) -> AsyncIterator[dict]:
    """Stream events as they occur."""
    async for event in event_source.subscribe(since_id):
        yield {
            "type": "event",
            "data": event.to_dict(),
            "id": event.id,
            "timestamp": event.timestamp.isoformat()
        }

# For MCP, wrap streaming in envelope when complete
@mcp.tool()
async def get_events_stream(
    since_id: Optional[str] = None,
    max_events: int = 100
) -> dict:
    """Get recent events with streaming support."""
    events = []
    async for event in stream_events(since_id):
        events.append(event)
        if len(events) >= max_events:
            break

    return asdict(success_response(
        data={
            "events": events,
            "streaming": True
        },
        pagination={
            "cursor": events[-1]["id"] if events else None,
            "has_more": len(events) >= max_events
        }
    ))
```

### Partial Result Indication

```json
{
    "success": true,
    "data": {
        "events": [...],
        "partial": true,
        "reason": "max_events_reached"
    },
    "meta": {
        "version": "response-v2",
        "warnings": ["Results truncated at 1000 events. Use cursor to continue."]
    }
}
```

## Idempotency

### Idempotency Levels

| Level | Description | Example |
|-------|-------------|---------|
| **Naturally Idempotent** | Safe to retry any number of times | GET, read operations |
| **Idempotent with Key** | Same key produces same result | POST with idempotency key |
| **Not Idempotent** | Each call has different effect | Increment counter |

### Document Idempotency

```python
@mcp.tool()
def get_user(user_id: str) -> dict:
    """Fetch a user by ID.

    Idempotency: Naturally idempotent. Safe to retry.
    """
    ...

@mcp.tool()
def create_order(
    items: List[dict],
    idempotency_key: str
) -> dict:
    """Create a new order.

    Idempotency: Idempotent when same idempotency_key is provided.
    Duplicate requests with same key return original result.

    Args:
        items: Order items
        idempotency_key: Client-generated unique key for deduplication
    """
    ...

@mcp.tool()
def increment_counter(counter_id: str) -> dict:
    """Increment a counter.

    Idempotency: NOT idempotent. Each call increments the counter.
    For safe retries, use increment_counter_idempotent with a key.
    """
    ...
```

### Idempotency Key Implementation

```python
from datetime import datetime, timedelta
import hashlib

IDEMPOTENCY_TTL = timedelta(hours=24)

@mcp.tool()
def create_payment(
    amount: float,
    recipient: str,
    idempotency_key: str
) -> dict:
    """Create a payment with idempotency support.

    Args:
        amount: Payment amount
        recipient: Recipient identifier
        idempotency_key: Unique key for this payment attempt
    """
    # Check for existing result
    existing = idempotency_store.get(idempotency_key)
    if existing:
        if existing["expires_at"] > datetime.utcnow():
            # Return cached result
            return existing["response"]
        else:
            # Expired, clean up
            idempotency_store.delete(idempotency_key)

    # Process payment
    try:
        result = payment_processor.create(amount, recipient)

        response = asdict(success_response(
            data={"payment_id": result.id, "status": result.status}
        ))

        # Cache result
        idempotency_store.set(
            idempotency_key,
            {
                "response": response,
                "expires_at": datetime.utcnow() + IDEMPOTENCY_TTL
            }
        )

        return response

    except PaymentError as e:
        # Also cache errors for idempotency
        response = asdict(error_response(str(e)))
        idempotency_store.set(
            idempotency_key,
            {
                "response": response,
                "expires_at": datetime.utcnow() + IDEMPOTENCY_TTL
            }
        )
        return response
```

### Client-Side Idempotency Key Generation

```python
import uuid
import hashlib

# Option 1: Random UUID (simple)
idempotency_key = str(uuid.uuid4())

# Option 2: Content-based (deterministic)
def generate_idempotency_key(operation: str, params: dict) -> str:
    """Generate deterministic idempotency key from parameters."""
    content = f"{operation}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]
```

## Anti-Patterns

### Don't: Expose Implementation in Cursors

```python
# Bad: Cursor exposes database offset
cursor = "offset_500"  # Reveals pagination implementation

# Good: Opaque cursor
cursor = "eyJsYXN0X2lkIjogIml0ZW1fNTAwIn0="  # Base64 encoded, opaque
```

### Don't: Skip Idempotency Documentation

```python
# Bad: No idempotency info
@mcp.tool()
def send_email(to: str, body: str) -> dict:
    """Send an email."""
    ...

# Good: Clear idempotency documentation
@mcp.tool()
def send_email(to: str, body: str, idempotency_key: str = None) -> dict:
    """Send an email.

    Idempotency: Without key, NOT idempotent (sends duplicate emails).
    With idempotency_key, same email sent only once per key.
    """
    ...
```

### Don't: Return Inconsistent Pagination

```python
# Bad: Different pagination formats
# Tool A: {"next_page": 2}
# Tool B: {"cursor": "..."}
# Tool C: {"offset": 100}

# Good: Consistent format across all tools
{"pagination": {"cursor": "...", "has_more": true}}
```

## Related Documents

- [Envelopes & Metadata](./02-envelopes-metadata.md) - Pagination in meta
- [Error Semantics](./07-error-semantics.md) - Handling pagination errors
- [Resilience Patterns](./12-timeout-resilience.md) - Safe retries

---

**Navigation:** [← Observability & Telemetry](./05-observability-telemetry.md) | [Index](./README.md) | [Next: Error Semantics →](./07-error-semantics.md)
