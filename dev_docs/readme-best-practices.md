# 19. README Best Practices

> Guidance derived from the README_examples set to help write clear, persuasive, and usable READMEs.

This document uses RFC 2119 terminology. See [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119) for definitions of MUST, SHOULD, and MAY.

---

## Overview

The README_examples set spans 17 CLI tools, plugins, and multi-agent frameworks. Despite differences in scope, they converge on a predictable structure: clear title and value statement, a short feature summary, installation/setup, usage workflows, and credibility signals (metrics, comparisons, licenses, and status).

This document captures the shared practices and turns them into a repeatable template for writing our own README.

---

## Common Sections and Purposes

| Section | Purpose | When to include |
|---------|---------|----------------|
| **Title + One-liner** | Names the product and states the primary benefit | Always |
| **Overview / Background / Problem** | Explains why the tool exists and who it helps | When the tool replaces a workflow or solves a clear pain |
| **Key Features** | Scannable list of capabilities and benefits | Always |
| **Installation / Setup / Prerequisites** | Gets users running with minimal friction | Always |
| **Usage / Example Flow** | Shows typical commands and expected outcomes | Always for CLI/tools |
| **Configuration** | Documents env vars, flags, and defaults | When configuration is required |
| **How It Works / Architecture** | Builds trust via clarity on internals | When behavior is non-obvious |
| **Commands / API Reference** | Single source of truth for usage | When there are many commands |
| **Advanced Features** | Highlights optional power-user capabilities | When there is depth beyond basics |
| **Limitations / Scope** | Sets expectations and avoids mis-use | When data sources or coverage are limited |
| **Comparison** | Differentiates from alternatives | When the space is crowded |
| **Support / Community / License** | Operational and legal clarity | Always |

---

## Recommended Ordering

### Short README (single-page)

1. Title + one-liner
2. 2-4 sentence overview (problem and outcome)
3. Key features (5-8 bullets)
4. Install / prerequisites
5. Quick start
6. Usage example
7. Configuration (if needed)
8. License + support

### Long README (multi-section)

1. Title + one-liner
2. Overview / Background
3. Table of contents
4. Key features
5. Installation / Setup
6. Quick start
7. Usage / Example flow
8. Architecture / How it works
9. Commands / API reference
10. Advanced features
11. Scope / limitations
12. Comparison
13. Support / community
14. License

Use a table of contents if the README exceeds ~8 sections or ~150 lines.

---

## Writing Style Patterns That Work

- **Benefit-first framing.** Open with the outcome, not the implementation.
- **Problem-solution narrative.** Explain the pain, then the fix.
- **Short paragraphs + bold labels.** Use `**Feature**: description` to scan quickly.
- **Specificity beats adjectives.** Numbers, scope, or limits make claims credible.
- **Actionable language.** Use verbs and direct commands in instructions.
- **Consistent naming.** Keep terms and command names identical across sections.
- **Structured examples.** Use commented code blocks to explain steps.
- **Principles as quotes.** A single short quote can anchor philosophy or guardrails.

Avoid vague phrasing like "powerful" or "comprehensive" without concrete support.

---

## Section Guidance

### Title and One-liner

- MUST include the exact product name.
- SHOULD include a descriptor (CLI, plugin, framework, etc.).
- MAY include badges (status, license, install).

Example style:
```
# ProductName - Short Descriptor
One sentence on what it does and who it is for.
```

### Overview / Background

- SHOULD name the user and the problem being solved.
- SHOULD state the primary outcome (time saved, clarity gained, etc.).
- MAY include short "why now" or origin context if it builds trust.

### Key Features

- SHOULD list 5-8 items max.
- SHOULD use short bold labels, then 1-2 sentences of detail.
- SHOULD mix feature and benefit (what + why it matters).

### Installation and Prerequisites

- MUST list required runtimes and minimum versions.
- SHOULD offer the simplest install first, with alternatives after.
- SHOULD keep commands copy-pastable and minimal.

### Quick Start

- SHOULD be 3-5 steps.
- SHOULD use a numbered list with one code block per step.
- SHOULD show a working default (no extra configuration).

### Usage / Example Flow

- MUST include at least one full workflow for CLI tools.
- SHOULD include inline comments to explain steps.
- MAY include outputs if they are short and non-noisy.

### Configuration

- SHOULD describe env vars and flags in a table.
- SHOULD document defaults and expected values.
- MAY include a sample config snippet.

### Architecture / How It Works

- SHOULD use a simple flow diagram or phase list.
- MAY include a directory tree for repo layout.

### Commands / API Reference

- SHOULD group commands by theme (setup, workflow, maintenance).
- SHOULD keep each line to one purpose and one sentence.

### Scope and Limitations

- SHOULD call out known constraints (data sources, platforms).
- SHOULD clarify "best for" and "not for" cases.

### Comparison

- SHOULD be factual and specific.
- SHOULD avoid "better than X" without a reason.

### Support, Community, License

- MUST state the license.
- SHOULD point to primary support channel (issues, Discord, email).
- MAY include contribution guidelines or status/roadmap.

---

## Formatting and Visual Patterns

- Use tables for comparisons, tool matrices, or local vs remote behavior.
- Use fenced code blocks with language tags (bash, yaml, json).
- Keep headings consistent (Overview, Key Features, Installation).
- For long docs, use a short list-based table of contents near the top.

---

## README Quality Checklist

- **MUST** state what the project is and who it is for in the first 3 lines.
- **MUST** include installation steps and a working example.
- **MUST** document prerequisites and minimum versions.
- **MUST** include license and support information.
- **SHOULD** include a key features list with benefits.
- **SHOULD** include at least one example workflow.
- **SHOULD** define scope and limitations if the tool is not universal.
- **MAY** include architecture diagrams, comparisons, and metrics.

---

## Suggested README Skeleton

```
# Product Name - Short Descriptor
One sentence describing the value and audience.

## Overview
2-4 sentences on the problem and outcome.

## Key Features
- **Feature**: Benefit-focused explanation.
- **Feature**: Benefit-focused explanation.

## Installation
Prerequisites and install commands.

## Quick Start
1. Step
2. Step
3. Step

## Usage
Example workflow with commands and comments.

## Configuration
Table or short list of env vars and defaults.

## How It Works
Short flow or architecture diagram.

## Commands
Grouped list of commands with one-line descriptions.

## Limitations
What this tool is not suited for.

## Support
Where to ask questions or file issues.

## License
License name and link.
```
