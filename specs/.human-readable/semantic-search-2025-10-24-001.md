# Semantic Search for Documentation Query

**Spec ID:** `semantic-search-2025-10-24-001`  
**Status:** pending (0/62 tasks, 0%)  
**Estimated Effort:** 80 hours  
**Complexity:** high  

Three-tier hybrid semantic search system combining keyword (regex), BM25 (fast keyword ranking), and semantic embeddings (conceptual understanding)

## Objectives

- Enable semantic search for natural language code queries
- Improve keyword search relevance with BM25
- Achieve <50ms BM25 search for 1000 entities
- Achieve <500ms semantic search for 1000 entities
- Zero breaking changes for existing users

## Foundation & Configuration (0/8 tasks, 0%)

**Purpose:** Set up dependencies, configuration modules, and code-aware tokenization  
**Risk Level:** low  
**Estimated Hours:** 6  


### File Modifications (0/6 tasks)

#### ⏳ src/claude_skills/pyproject.toml

**File:** `src/claude_skills/pyproject.toml`  
**Status:** pending  
**Estimated:** 0.5 hours  
**Changes:** Add optional dependency groups for semantic search  
**Reasoning:** Enable progressive enhancement - users can opt-in to BM25 and/or semantic features  

##### ⏳ Add [semantic] optional dependency group

**Status:** pending  

**Details:** Add rank-bm25>=0.2.2, sentence-transformers>=2.2.0, numpy>=1.21.0 to [project.optional-dependencies]

##### ⏳ Add [semantic-cpu] optional dependency group

**Status:** pending  

**Details:** Same as semantic but with CPU-only torch for non-Mac platforms

#### ⏳ src/claude_skills/claude_skills/doc_query/semantic_config.py

**File:** `src/claude_skills/claude_skills/doc_query/semantic_config.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Create new configuration module for semantic search  
**Reasoning:** Centralize configuration for BM25, embeddings, and search behavior  

##### ⏳ Create BM25Config dataclass

**Status:** pending  

**Details:** k1=1.5, b=0.75, epsilon=0.25 (BM25 algorithm parameters)

##### ⏳ Create EmbeddingConfig dataclass

**Status:** pending  

**Details:** model_name='all-MiniLM-L6-v2', batch_size=32, max_seq_length=384

##### ⏳ Create SearchConfig dataclass

**Status:** pending  

**Details:** bm25_top_k=50, semantic_top_k=10, similarity_threshold=0.3

##### ⏳ Implement tokenize_code_aware() function

**Status:** pending  

**Details:** Handle snake_case, camelCase, PascalCase splitting. Use minimal stopwords (only 'the', 'a', 'an', 'and', 'or', 'but') - keep 'in', 'on', 'at' for code context


### Verification (0/2 tasks)

**Blocked by:** phase-1-files  

#### ⏳ Tokenization handles code patterns correctly

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/test_semantic_config.py::test_tokenize_code_aware
```

**Expected:** getUserName → ['get', 'user', 'name'], HTTPServer → ['http', 'server'], parse_json_file → ['parse', 'json', 'file']

#### ⏳ Configuration dataclasses have correct defaults

**Status:** pending  
**Type:** manual  

**Expected:** BM25Config, EmbeddingConfig, SearchConfig instantiate with documented defaults


## BM25 Search Implementation (0/11 tasks, 0%)

**Purpose:** Add fast BM25-based search as first enhancement tier  
**Risk Level:** low  
**Estimated Hours:** 8  

**Blocked by:** phase-1  

### File Modifications (0/8 tasks)

#### ⏳ src/claude_skills/claude_skills/doc_query/bm25_search.py

**File:** `src/claude_skills/claude_skills/doc_query/bm25_search.py`  
**Status:** pending  
**Estimated:** 3 hours  
**Changes:** Create new BM25 search module  
**Reasoning:** Provides improved keyword search with better relevance ranking  

**Depends on:** task-1-2

##### ⏳ Create BM25Result dataclass

**Status:** pending  

**Details:** entity_id, score, entity_data fields

##### ⏳ Create BM25Index class with build() method

**Status:** pending  

**Details:** Use rank-bm25 library, tokenize with tokenize_code_aware(), weight name higher (2x repetition)

##### ⏳ Implement BM25Index.search() method

**Status:** pending  

**Details:** Return top-k results sorted by BM25 score

##### ⏳ Add check_bm25_available() helper

**Status:** pending  

**Details:** Gracefully handle ImportError if rank-bm25 not installed

#### ⏳ src/claude_skills/claude_skills/doc_query/doc_query_lib.py

**File:** `src/claude_skills/claude_skills/doc_query/doc_query_lib.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Integrate BM25 indices into DocumentationQuery  
**Reasoning:** Enable BM25 search within existing query interface  

**Depends on:** task-2-1

##### ⏳ Add BM25 state to __init__()

**Status:** pending  

**Details:** self._bm25_available, self._bm25_indices = {classes, functions, modules}

##### ⏳ Create _build_bm25_indices() method

**Status:** pending  

**Details:** Build BM25Index for classes, functions, modules with error handling

##### ⏳ Call _build_bm25_indices() in _reindex()

**Status:** pending  

**Details:** Build BM25 indices after standard reindexing if available

##### ⏳ Add error handling for BM25 build failures

**Status:** pending  

**Details:** Disable BM25 and log warning if build fails, don't crash


### Verification (0/3 tasks)

**Blocked by:** phase-2-files  

#### ⏳ BM25 search returns relevant results

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/test_bm25_search.py::test_bm25_search
```

**Expected:** Query 'user login' returns authenticate_user and login_handler in top results

#### ⏳ BM25 gracefully degrades when not installed

**Status:** pending  
**Type:** manual  

**Expected:** DocumentationQuery works without rank-bm25, _bm25_available=False

#### ⏳ BM25 search performance meets target

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/test_bm25_search.py::test_bm25_performance
```

**Expected:** BM25 search completes in <50ms for 1000 entities


## Semantic Embeddings Infrastructure (0/19 tasks, 0%)

**Purpose:** Enable embedding generation for semantic search  
**Risk Level:** medium  
**Estimated Hours:** 14  

**Blocked by:** phase-1  

### File Modifications (0/15 tasks)

#### ⏳ src/claude_skills/claude_skills/doc_query/embeddings.py

**File:** `src/claude_skills/claude_skills/doc_query/embeddings.py`  
**Status:** pending  
**Estimated:** 5 hours  
**Changes:** Create embedding generation module in doc_query (NOT code_doc)  
**Reasoning:** Embeddings used by doc_query, not code_doc. Includes all critical fixes from review.  

**Depends on:** task-1-2

##### ⏳ Add imports including 'from datetime import datetime'

**Status:** pending  

**Details:** CRITICAL FIX: Missing import caused failure. Import json, numpy, datetime, pathlib, typing, sentence_transformers, semantic_config

##### ⏳ Create EmbeddingResult dataclass

**Status:** pending  

**Details:** embeddings, metadata, model_name, embedding_dim fields

##### ⏳ Create EmbeddingGenerator class with lazy model loading

**Status:** pending  

**Details:** Lazy @property for model loading. Add try/except with helpful error messages for network issues, corrupted cache, disk space

##### ⏳ Implement generate_for_entities() method

**Status:** pending  

**Details:** Generate embeddings for list of entities with progress bar support (tqdm)

##### ⏳ Implement _create_embedding_text() static method

**Status:** pending  

**Details:** Create searchable text from entity: type, name, docstring, file context, parameters, inheritance

##### ⏳ Implement save_embeddings() method with staleness tracking

**Status:** pending  

**Details:** Save as NPZ (compressed numpy) + metadata JSON. Include model_version, embedding_version, codebase_hash for staleness detection

##### ⏳ Add check_embeddings_available() and compute_codebase_hash() helpers

**Status:** pending  

**Details:** Check if sentence-transformers available. Compute SHA-256 hash of documentation.json for staleness detection

#### ⏳ src/claude_skills/claude_skills/code_doc/cli.py

**File:** `src/claude_skills/claude_skills/code_doc/cli.py`  
**Status:** pending  
**Estimated:** 3 hours  
**Changes:** Add embedding generation to doc generation workflow  
**Reasoning:** Allow users to generate embeddings during initial doc generation  

**Depends on:** task-3-1

##### ⏳ Add --generate-embeddings flag to generate command

**Status:** pending  

**Details:** Boolean flag to enable embedding generation during doc generation

##### ⏳ Update cmd_generate() to call EmbeddingGenerator

**Status:** pending  

**Details:** After doc generation, generate embeddings for classes, functions, modules if flag set. Show progress bars. Handle ImportError gracefully.

##### ⏳ Update documentation.json metadata with embeddings info

**Status:** pending  

**Details:** Add metadata.embeddings = {available: true, model: ..., path: 'embeddings.npz', dimension: 384}

#### ⏳ src/claude_skills/claude_skills/code_doc/cli.py (generate-embeddings command)

**File:** `src/claude_skills/claude_skills/code_doc/cli.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add new generate-embeddings subcommand  
**Reasoning:** Allow users to generate embeddings for existing documentation  

**Depends on:** task-3-1

##### ⏳ Create generate-embeddings subparser

**Status:** pending  

**Details:** Add parser with docs_path (optional), --model (default: all-MiniLM-L6-v2), --force flags

##### ⏳ Implement cmd_generate_embeddings() function

**Status:** pending  

**Details:** Load existing documentation.json, create EmbeddingGenerator, generate embeddings

##### ⏳ Add staleness check

**Status:** pending  

**Details:** Check if embeddings exist and are up-to-date using codebase_hash. Skip if current unless --force

##### ⏳ Show progress bars during generation

**Status:** pending  

**Details:** Use tqdm to show progress for embedding generation (can take 5-10s for 500 entities)

##### ⏳ Handle errors gracefully

**Status:** pending  

**Details:** ImportError if sentence-transformers missing, file I/O errors, model loading failures. Show helpful install instructions.


### Verification (0/4 tasks)

**Blocked by:** phase-3-files  

#### ⏳ Embeddings generate successfully

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/test_embeddings.py::test_embedding_generation
```

**Expected:** Embeddings array shape (N, 384) for all-MiniLM-L6-v2 model

#### ⏳ Embedding files saved correctly

**Status:** pending  
**Type:** manual  

**Expected:** embeddings.npz and embeddings_meta.json exist, metadata includes model, dimension, codebase_hash, entity counts

#### ⏳ Staleness detection works

**Status:** pending  
**Type:** manual  

**Expected:** Modify documentation.json, check that codebase_hash changes and embeddings marked stale

#### ⏳ Error handling works for model loading

**Status:** pending  
**Type:** manual  

**Expected:** Simulate network failure, verify helpful error message about connectivity, corrupted cache, disk space


## Hybrid Search Orchestration (0/18 tasks, 0%)

**Purpose:** Combine BM25 and semantic search for optimal results  
**Risk Level:** medium  
**Estimated Hours:** 12  

**Blocked by:** phase-2, phase-3  

### File Modifications (0/14 tasks)

#### ⏳ src/claude_skills/claude_skills/doc_query/semantic_search.py

**File:** `src/claude_skills/claude_skills/doc_query/semantic_search.py`  
**Status:** pending  
**Estimated:** 3 hours  
**Changes:** Create semantic search module using embeddings  
**Reasoning:** Enable pure semantic search based on cosine similarity  

**Depends on:** task-3-1

##### ⏳ Create SemanticResult dataclass

**Status:** pending  

**Details:** entity_id, entity_type, score, entity_data fields

##### ⏳ Create SemanticIndex class with load() method

**Status:** pending  

**Details:** Load embeddings.npz and metadata, load SentenceTransformer model

##### ⏳ Implement SemanticIndex.search() method

**Status:** pending  

**Details:** Generate query embedding, compute cosine similarity, filter by threshold, return top-k

##### ⏳ Add _cosine_similarity() static method

**Status:** pending  

**Details:** Efficient numpy-based cosine similarity computation

#### ⏳ src/claude_skills/claude_skills/doc_query/hybrid_search.py

**File:** `src/claude_skills/claude_skills/doc_query/hybrid_search.py`  
**Status:** pending  
**Estimated:** 4 hours  
**Changes:** Create hybrid search orchestrator combining BM25 + semantic  
**Reasoning:** Provides best search quality by combining fast BM25 filtering with accurate semantic ranking  

**Depends on:** task-2-1, task-4-1

##### ⏳ Create SearchStrategy enum

**Status:** pending  

**Details:** KEYWORD, BM25, SEMANTIC, HYBRID, AUTO values

##### ⏳ Create HybridSearchEngine class with capability detection

**Status:** pending  

**Details:** _detect_capabilities() returns {keyword, bm25, semantic} availability

##### ⏳ Implement strategy selection and search routing

**Status:** pending  

**Details:** search() method routes to _keyword_search(), _bm25_search(), _semantic_search(), or _hybrid_search() based on strategy

##### ⏳ Implement _hybrid_search() with min-max normalization

**Status:** pending  

**Details:** CRITICAL FIX: BM25 filter (top 50) → semantic re-rank. Use min-max normalization for BM25 scores, not division by 10. Combine: 0.4*bm25_norm + 0.6*semantic

##### ⏳ Add get_status() method for capability reporting

**Status:** pending  

**Details:** Return capabilities, recommended_strategy, available_strategies for CLI display

#### ⏳ src/claude_skills/claude_skills/doc_query/doc_query_lib.py (semantic integration)

**File:** `src/claude_skills/claude_skills/doc_query/doc_query_lib.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Integrate semantic search into DocumentationQuery  
**Reasoning:** Enable semantic search alongside BM25  

**Depends on:** task-4-1

##### ⏳ Add semantic index loading to __init__()

**Status:** pending  

**Details:** Check for embeddings, create SemanticIndex, call load(). Set self._semantic_index

##### ⏳ Update search_entities() to use HybridSearchEngine

**Status:** pending  

**Details:** Add strategy parameter (default 'auto'), create HybridSearchEngine, delegate to engine.search()

#### ⏳ src/claude_skills/claude_skills/doc_query/cli.py

**File:** `src/claude_skills/claude_skills/doc_query/cli.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add --strategy argument to search commands  
**Reasoning:** Allow users to select search strategy explicitly  

**Depends on:** task-4-2

##### ⏳ Add --strategy argument to parser

**Status:** pending  

**Details:** choices=['keyword', 'bm25', 'semantic', 'hybrid', 'auto'], default='auto'

##### ⏳ Update cmd_search() to pass strategy

**Status:** pending  

**Details:** Call query.search_entities(args.query, strategy=args.strategy). Show which strategy was used.

##### ⏳ Update cmd_stats() to show search capabilities

**Status:** pending  

**Details:** Use HybridSearchEngine.get_status() to show available strategies, recommended strategy, installation instructions


### Verification (0/4 tasks)

**Blocked by:** phase-4-files  

#### ⏳ Semantic search finds conceptually similar results

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/test_semantic_search.py::test_semantic_similarity
```

**Expected:** Query 'user authentication logic' finds authenticate_user(), LoginHandler even without exact text match

#### ⏳ Hybrid search combines BM25 and semantic correctly

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/test_hybrid_search.py::test_hybrid_reranking
```

**Expected:** Hybrid achieves better ranking than BM25 or semantic alone. Min-max normalization works correctly.

#### ⏳ Strategy auto-selection works

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/test_hybrid_search.py::test_strategy_selection
```

**Expected:** AUTO selects HYBRID when semantic available, BM25 when only BM25 available, KEYWORD as fallback

#### ⏳ Graceful degradation works

**Status:** pending  
**Type:** manual  

**Expected:** Request semantic search without embeddings → falls back to BM25 with warning. Request BM25 without rank-bm25 → falls back to keyword with warning.


## Testing, Documentation & Polish (0/6 tasks, 0%)

**Purpose:** Comprehensive testing, UX improvements, and documentation  
**Risk Level:** low  
**Estimated Hours:** 16  

**Blocked by:** phase-1, phase-2, phase-3, phase-4  

### File Modifications (0/3 tasks)

#### ⏳ src/claude_skills/tests/test_*.py (comprehensive test suite)

**File:** `src/claude_skills/tests/`  
**Status:** pending  
**Estimated:** 8 hours  
**Changes:** Create comprehensive test suite for all new modules  
**Reasoning:** Ensure code quality, prevent regressions, validate performance targets  

**Details:** test_semantic_config.py (tokenization), test_bm25_search.py (BM25 + performance), test_embeddings.py (generation + staleness), test_semantic_search.py (similarity), test_hybrid_search.py (orchestration + strategy), test_integration.py (end-to-end workflow). Target >80% coverage.

#### ⏳ src/claude_skills/claude_skills/doc_query/SKILL.md

**File:** `src/claude_skills/claude_skills/doc_query/SKILL.md`  
**Status:** pending  
**Estimated:** 3 hours  
**Changes:** Update doc-query skill documentation with semantic search info  
**Reasoning:** Users need to understand new capabilities and how to use them  

**Details:** Add sections: Search Strategies, Installation, Generating Embeddings, Usage Examples, When to Use Each Strategy, Performance Characteristics

#### ⏳ docs/SEMANTIC_SEARCH_MIGRATION.md

**File:** `docs/SEMANTIC_SEARCH_MIGRATION.md`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Create migration guide for existing users  
**Reasoning:** Help users adopt semantic search with minimal friction  

**Details:** No breaking changes section, how to enable semantic search, performance expectations, troubleshooting guide


### Verification (0/3 tasks)

**Blocked by:** phase-5-files  

#### ⏳ All tests pass with >80% coverage

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/ --cov=claude_skills.doc_query --cov-report=term-missing
```

**Expected:** All tests pass, coverage >80% for semantic_config, bm25_search, embeddings, semantic_search, hybrid_search

#### ⏳ Performance targets met

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/test_performance.py
```

**Expected:** BM25 <50ms, Semantic <500ms (after first load), Hybrid <300ms for 1000 entities

#### ⏳ No breaking changes for existing users

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest tests/test_backward_compatibility.py
```

**Expected:** All existing sdd doc commands work without installing new dependencies. Default behavior unchanged.
