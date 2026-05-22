# 🔍 AI Code Review Agent

An autonomous AI-powered code review agent that clones GitHub repositories, analyzes code using Abstract Syntax Tree (AST) parsing, and generates confidence-rated review comments via GPT-4o-mini.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 🎯 Overview

This project implements a fully functional agentic AI pipeline that:

1. **Clones** any public GitHub repository via GitPython
2. **Parses** Python source files using AST to extract functions, classes, methods, and imports
3. **Chunks** large files intelligently respecting AST boundaries
4. **Reviews** code using GPT-4o-mini with engineered prompts for consistent structured output
5. **Rates** every finding with a confidence score (0–100%)
6. **Displays** results in a polished Streamlit dashboard with filters, charts, and exports

### ✨ Key Feature: Epistemic Humility

Every review comment includes a self-rated **confidence score**. Low-confidence comments (< 50%) are displayed in a separate "⚠️ Needs Verification" section with visual indicators — demonstrating production-grade uncertainty quantification.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                       │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐  │
│  │ Filters  │  │  Charts  │  │  Cards  │  │   Exports    │  │
│  └────┬─────┘  └────┬─────┘  └────┬────┘  └──────┬───────┘  │
│       └──────────────┴────────────┴───────────────┘          │
├──────────────────────────┬───────────────────────────────────┤
│       Orchestrator       │        Pipeline Flow              │
│                          │                                   │
│   ┌─────────────┐        │   GitHub URL                      │
│   │  Ingestion  │────────┼──►  Clone (GitPython, depth=1)    │
│   └──────┬──────┘        │                                   │
│          ▼               │                                   │
│   ┌─────────────┐        │   .py files                       │
│   │  AST Parser │────────┼──►  Extract functions, classes     │
│   └──────┬──────┘        │     imports, complexity hints      │
│          ▼               │                                   │
│   ┌─────────────┐        │   Code chunks                     │
│   │   Chunker   │────────┼──►  Split at AST boundaries       │
│   └──────┬──────┘        │     with context preambles        │
│          ▼               │                                   │
│   ┌─────────────┐        │   Structured JSON                 │
│   │ LLM Reviewer│────────┼──►  GPT-4o-mini with              │
│   └──────┬──────┘        │     confidence scoring            │
│          ▼               │                                   │
│   ┌─────────────┐        │   ReviewReport                    │
│   │  Aggregator │────────┼──►  Pydantic-validated results    │
│   └─────────────┘        │                                   │
├──────────────────────────┴───────────────────────────────────┤
│                    Export Layer                               │
│         Markdown  │  CSV  │  JSON  │  GitHub PR API          │
└──────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ai-code-review-agent/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── config/
│   ├── __init__.py
│   └── settings.py             # Centralized configuration & constants
├── core/
│   ├── __init__.py
│   ├── models.py               # Pydantic data models
│   ├── ingestion.py            # GitPython repository cloning
│   ├── parser.py               # AST parsing engine
│   ├── chunker.py              # Intelligent code chunking
│   ├── reviewer.py             # LLM review with structured output
│   └── orchestrator.py         # Pipeline coordinator
├── utils/
│   ├── __init__.py
│   ├── export.py               # Markdown/CSV/JSON export
│   └── github_api.py           # GitHub PR comment posting (bonus)
└── assets/
    └── style.css               # Custom dark theme CSS
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Git installed and on PATH
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/ai-code-review-agent.git
cd ai-code-review-agent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run the application
streamlit run app.py
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for GPT-4o-mini |
| `GITHUB_TOKEN` | ❌ No | GitHub PAT for PR comment posting (bonus) |

## 🎮 Usage

1. Open the app in your browser (typically `http://localhost:8501`)
2. Paste a public GitHub repository URL in the sidebar
3. Enter your OpenAI API key (or set it in `.env`)
4. Click **🚀 Analyze Repository**
5. Watch the pipeline run in real-time with progress updates
6. Browse results filtered by category, severity, and confidence
7. Download reports in Markdown, CSV, or JSON format

## 🧠 How It Works

### AST Parsing
The agent uses Python's built-in `ast` module to extract:
- **Functions & async functions** — with argument counts, return statements, docstrings
- **Classes** — with base classes, method counts, total line spans
- **Methods** — class-scoped functions with complexity metrics
- **Imports** — module dependencies for context
- **Complexity hints** — nesting depth, bare excepts, argument counts

### Smart Chunking
Large files are split at AST boundaries (not arbitrary line counts):
- Each chunk includes a context preamble (imports + structure overview)
- Functions and classes are kept whole whenever possible
- Oversized elements are split at logical line boundaries

### Prompt Engineering
The system prompt enforces:
- Consistent JSON schema output via `response_format={"type": "json_object"}`
- Explicit confidence scoring guidelines (90–100 = certain, <50 = speculative)
- A "quality over quantity" directive to prevent hallucinated issues
- Pydantic validation on every response

### Confidence Scoring
Comments are bucketed into three tiers:
| Bucket | Range | Display |
|--------|-------|---------|
| 🟢 High | ≥ 70% | Shown prominently as reliable findings |
| 🟡 Medium | 50–69% | Shown with context caveat |
| 🔴 Low | < 50% | "⚠️ Verify This" label + dashed border |

## ⚠️ Known Limitations

1. **Python only** — Currently parses only `.py` files. JavaScript/Go support would require tree-sitter integration.
2. **Public repos only** — Private repository access requires GitHub token authentication (not implemented for cloning).
3. **File cap** — Analyzes a maximum of 50 files per repository to control API costs.
4. **Single-language AST** — Uses Python's `ast` module, which cannot parse other languages.
5. **No incremental review** — Analyzes the full repo each time; does not diff against previous runs.
6. **API costs** — Each review consumes OpenAI API tokens. Large repos may incur noticeable costs.

## 🔮 Future Improvements

With more time, I would build:

1. **Multi-language support** via tree-sitter (JavaScript, TypeScript, Go, Rust)
2. **Incremental diff analysis** — review only changed files between commits
3. **Custom rule configuration** — let users define which categories to focus on
4. **Caching layer** — avoid re-reviewing unchanged files
5. **GitHub Actions integration** — run as a CI/CD step on every PR
6. **Fine-tuned model** — train on real code review data for higher accuracy
7. **Security scanning** — integrate with CVE databases for dependency checks
8. **Multi-model consensus** — use multiple LLMs and aggregate confidence scores

## 📝 Tech Stack

| Component | Technology |
|-----------|-----------|
| Ingestion | GitPython |
| Parsing | Python `ast` module |
| LLM | OpenAI GPT-4o-mini |
| Orchestration | Custom Python pipeline |
| Dashboard | Streamlit |
| Data Models | Pydantic v2 |
| Charts | Plotly |
| Export | Markdown, CSV, JSON |
| PR Comments | PyGitHub (bonus) |

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for GPT-4o-mini
- [Streamlit](https://streamlit.io/) for the dashboard framework
- [GitPython](https://gitpython.readthedocs.io/) for repository management
- Test repositories used: Python standard library examples, open-source Flask/FastAPI projects

---

*Built with ❤️ as part of the CipherSchool AI Code Review Agent assignment.*
