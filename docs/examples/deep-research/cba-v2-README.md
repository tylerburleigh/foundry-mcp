# Deep Research Example: Conversation-Based Assessment

This contains an example output from the `deep-research` workflow, demonstrating how foundry-mcp conducts automated, multi-phase research on a topic.

## Research Query

> "conversation based assessment: methods, frameworks, best practices, applications in education and professional evaluation, AI-powered conversational assessment systems, validity and reliability considerations"

## Workflow Overview

The deep research workflow executes in distinct phases:

### Phase 1: Planning
The system analyzes the query and generates targeted sub-queries to explore different facets of the topic. For this research, it generated 12 sub-queries covering:
- Theoretical frameworks and methodologies
- Clinical and healthcare applications (cognitive assessment, mental health screening)
- Professional evaluation (recruitment, HR automation)
- Educational assessment (grading validity, reliability)
- Regulatory compliance (NYC Local Law 144, EU AI Act)
- Psychometric standards for AI (STAMP-LLM framework)

### Phase 2: Gathering
Each sub-query is executed against multiple search providers in parallel, yielding 70 unique sources.

### Content Digestion (PDF & HTML)

The workflow doesn't just collect URLs—it **fetches and digests full document content**, including PDFs. For each eligible source:

1. **Download** - Fetches the actual document (PDF or HTML)
2. **Extract** - Parses text content from the document
3. **Digest** - Compresses content using LLM summarization
4. **Index** - Extracts evidence snippets with relevance scores and character locators

Example from this research (ETS PDF source):
```
url: https://www.pt.ets.org/Media/Research/pdf/RD_Connections_25.pdf
content_type: digest/v1
original_chars: 21,654
digest_chars: 3,428
compression_ratio: 0.158 (15.8% of original)
_digest_duration_ms: 17,349
```

PDFs fetched in this research include:
- ETS Research: `RD_Connections_25.pdf` (Conversation-Based Assessment)
- ERIC Database: `EJ1476231.pdf` (Human vs AI Grading)
- NIST: `nist.ai.100-1.pdf` (AI Risk Management)
- ICWSM Proceedings: `2022_67.pdf` (Conversational Bias)
- SSRN Papers, academic PDFs from various universities

Source metadata tracks digestion status:
- `_digest_eligible`: Whether the source qualified for full processing
- `_digest_cache_hit`: Whether content was retrieved from cache
- `_digest_duration_ms`: Processing time for content extraction

### Phase 3: Analysis
Findings are synthesized, conflicts are identified, and knowledge gaps are noted for refinement iterations.

### Phase 4: Synthesis
A final report is generated with executive summary, key findings organized by theme, analysis of supporting/conflicting evidence, limitations, and actionable conclusions.

### Phase 5: Refinement
The workflow iterates up to 3 times, identifying gaps and generating additional sub-queries to fill them.

## Statistics

| Metric | Value |
|--------|-------|
| Total Iterations | 3 |
| Sub-queries Generated | 12 |
| Sub-queries Completed | 12 |
| Sources Examined | 70 |
| Sources Digested | 24 |
| PDFs Fetched | 8+ |
| Key Findings | 12 |
| Knowledge Gaps | 6 |
| Total Tokens Used | 222,403 |
| Duration | ~152 seconds |

## Files in This Directory

| File | Description |
|------|-------------|
| `conversation-based-assessment-report.md` | The final synthesized research report |
| `conversation-based-assessment-audit.jsonl` | Detailed audit trail of every operation (JSONL format) |
| `conversation-based-assessment-README.md` | This overview document |

## Usage

To run your own deep research:

```bash
# Start research (runs in background)
foundry research deep-research \
  --query "Your research topic here" \
  --max-iterations 3

# Check progress
foundry research deep-research-status --research-id <id>

# Get final report
foundry research deep-research-report --research-id <id>
```

Or via MCP tool calls:

```python
# Start
{"action": "deep-research", "query": "...", "max_iterations": 3}

# Status (shows live progress)
{"action": "deep-research-status", "research_id": "..."}

# Report
{"action": "deep-research-report", "research_id": "..."}
```

## Key Takeaways from This Research

The research revealed that conversation-based assessment is a transformative but complex paradigm:

1. **Diagnostic superiority** - CBA reveals mental models and reasoning processes invisible to static testing
2. **Efficiency gains** - AI-driven systems achieve 5-10x speed improvements and 10-25% cost reductions
3. **Clinical validation** - AI-administered cognitive and mental health assessments show reliability comparable to human-administered versions
4. **Validity gap in grading** - AI exhibits "score inflation" and lower inter-rater reliability vs. humans
5. **Emerging regulations** - NYC Local Law 144 and EU AI Act require bias audits and transparency notices
6. **New psychometric frameworks** - STAMP-LLM proposes standards specifically designed for evaluating AI "synthetic personalities"

## Recommendations

1. **Adopt Hybrid Models** - Keep humans in the loop for high-stakes decisions
2. **Standardize Audits** - Use frameworks like STAMP-LLM for AI-specific psychometric benchmarking
3. **Prioritize Compliance** - Implement bias audits and transparency notices from day one

## Source Diversity

The research drew from diverse domains including:
- Academic sources: arxiv.org, doi.org, pmc.ncbi.nlm.nih.gov, files.eric.ed.gov
- Assessment organizations: ETS (pt.ets.org)
- Industry: impress.ai, secondnature.ai, conveo.ai, fairly.ai
- Medical journals: Journal of Clinical and Experimental Neuropsychology
- Legal/regulatory: NYC Local Law 144 guides, EU AI Act analysis
