

# Blog Writing Agent

An AI-powered blog writing system built on **LangGraph** that takes a topic and autonomously researches, plans, writes, and illustrates a full technical blog post. It features a **Streamlit** frontend for interactive use and a modular **LangGraph** backend pipeline.

<img width="1440" height="900" alt="Screenshot 2026-04-02 at 2 28 08 PM" src="https://github.com/user-attachments/assets/f1d0ed30-a391-435a-8ecb-c9b2166237b9" />

<img width="1440" height="900" alt="Screenshot 2026-04-02 at 2 42 01 PM" src="https://github.com/user-attachments/assets/191d1ac8-876c-44c9-ad2d-9efd834c3c17" />

<img width="1440" height="900" alt="Screenshot 2026-04-02 at 2 41 36 PM" src="https://github.com/user-attachments/assets/1955def0-f4ed-494a-af71-eae52f8f9cce" />

<img width="1440" height="900" alt="Screenshot 2026-04-02 at 2 41 29 PM" src="https://github.com/user-attachments/assets/12e31d33-b189-44d4-9ca5-72c0225da9c8" />




---

## Features

- Automatic topic routing: decides whether web research is needed
- Real-time web research via **Tavily Search**
- Structured blog planning (title, audience, tone, per-section tasks)
- Sequential section writing with citation support
- AI image generation via **HuggingFace FLUX.1-schnell**
- Streamlit UI with live progress, tabbed output, and past-blog browser
- Downloads: markdown file, or full bundle (markdown + images) as ZIP

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Web Research | Tavily Search API |
| Image Generation | HuggingFace Inference — `FLUX.1-schnell` |
| Frontend | Streamlit |
| Structured Output | Pydantic v2 + LangChain |

---

## Project Structure

```
Blog-writing-Agent/
├── bwa_backend.py          # LangGraph pipeline (all nodes + graph)
├── bwa_frontend.py         # Streamlit UI
├── 1_bwa.ipynb             # Notebook: basic agent prototype
├── 2_bwa_research_fine_tuned.ipynb  # Notebook: research-enhanced version
├── 3_bwa_image.ipynb       # Notebook: image generation integration
├── tavilysearch_test.ipynb # Notebook: Tavily search experiments
├── images/                 # Generated images saved here
├── *.md                    # Generated blog posts saved here
├── .env                    # API keys (not committed)
└── .gitignore
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Blog-writing-Agent
```

### 2. Install dependencies

```bash
pip install streamlit langgraph langchain langchain-groq langchain-community \
            pydantic python-dotenv pandas requests
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key      # optional: enables web research
HF_TOKEN=your_huggingface_token         # optional: enables image generation
```

- **GROQ_API_KEY** (required): Get one at [console.groq.com](https://console.groq.com)
- **TAVILY_API_KEY** (optional): Get one at [tavily.com](https://tavily.com) — enables `hybrid` and `open_book` research modes
- **HF_TOKEN** (optional): Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — enables AI image generation

### 4. Run the app

```bash
streamlit run bwa_frontend.py
```

---

## Workflow

The backend is a compiled LangGraph `StateGraph`. Each blog generation runs through the following nodes:

```
START
  |
  v
[router]
  |
  +-- needs_research=True  --> [research] --> [orchestrator]
  |
  +-- needs_research=False ------------>  [orchestrator]
                                               |
                                          [worker_loop]
                                               |
                                          [reducer subgraph]
                                          /    |    \
                               merge_content   |   generate_and_place_images
                                         decide_images
                                               |
                                             END
```

### Node Descriptions

#### 1. Router
Analyzes the topic and decides the research strategy:

| Mode | Description | Recency Window |
|---|---|---|
| `closed_book` | Evergreen/conceptual topics — no research needed | N/A |
| `hybrid` | Mix of evergreen content + recent examples/tools | Last 45 days |
| `open_book` | News, weekly roundups, rapidly-changing topics | Last 7 days |

Outputs: `mode`, `needs_research`, `queries[]`, `recency_days`

#### 2. Research _(conditional)_
Runs only when `needs_research=True`. Executes each query against the Tavily Search API, then uses the LLM to synthesize raw results into structured `EvidenceItem` objects (title, URL, snippet, published date). Deduplicates by URL and filters by recency for `open_book` mode.

Outputs: `evidence[]`

#### 3. Orchestrator
Generates a structured `Plan` using the topic, mode, and evidence:
- Blog title, audience, tone, blog kind (`explainer`, `tutorial`, `news_roundup`, `comparison`, `system_design`)
- 5–9 `Task` objects, each with: goal, bullet points, target word count, flags (`requires_research`, `requires_citations`, `requires_code`)

Outputs: `plan`

#### 4. Worker Loop
Iterates through every task in the plan sequentially (with a 3-second delay between calls to respect rate limits). For each task, calls `worker_node` which instructs the LLM to write one markdown section, respecting:
- Target word count (±15%)
- All bullet points in order
- Citation rules (only cite provided Evidence URLs)
- Code snippet requirement if `requires_code=True`

Outputs: `sections[]` — list of `(task_id, markdown_text)` tuples

#### 5. Reducer Subgraph
A nested 3-node subgraph that assembles and illustrates the blog:

**merge_content**
Sorts sections by task ID and joins them under the blog title heading into a single `merged_md` string.

**decide_images**
The LLM reads the full merged markdown and decides where images would improve comprehension (max 3). It inserts `[[IMAGE_1]]`, `[[IMAGE_2]]`, `[[IMAGE_3]]` placeholders and writes detailed FLUX diffusion prompts for each, specifying layout, components, labels, and style.

**generate_and_place_images**
For each image spec, calls the HuggingFace Inference API (`FLUX.1-schnell`) to generate the image, saves it to `images/<filename>.png`, and replaces the placeholder with a markdown image reference. Falls back gracefully with a blockquote if generation fails. Saves the final markdown to disk.

Outputs: `md_with_placeholders`, `image_specs[]`, `final` (complete markdown)

---

## Frontend UI

The Streamlit app (`bwa_frontend.py`) provides:

**Sidebar**
- Topic text area + as-of date picker
- "Generate Blog" button
- Past blogs browser: lists all `.md` files in the working directory; clicking "Load selected blog" restores it into the UI without re-running the pipeline

**Main area tabs**

| Tab | Contents |
|---|---|
| Plan | Blog title, audience, tone, blog kind; interactive task table with per-task metadata |
| Evidence | Table of research sources (title, URL, published date, source) |
| Markdown Preview | Rendered blog with local images displayed inline; download buttons for `.md` and `.zip` bundle |
| Images | Image generation plan (JSON) + all generated images; download as ZIP |
| Logs | Raw streaming event log from the LangGraph run |

---

## Output Files

Each run saves two artifacts to the working directory:

- `<blog_title_slug>.md` — the complete blog post in Markdown
- `images/<filename>.png` — one file per generated image

Generated blogs persist across sessions and appear in the sidebar's past blogs list.

---

## Environment Variables Reference

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | LLM inference (routing, planning, writing, image planning) |
| `TAVILY_API_KEY` | No | Web research for `hybrid` and `open_book` topics |
| `HF_TOKEN` | No | Image generation via FLUX.1-schnell on HuggingFace |

Without `TAVILY_API_KEY`, the agent always runs in `closed_book` mode.
Without `HF_TOKEN`, image placeholders are replaced with descriptive fallback blockquotes instead of actual images.

---

## Example Topics

| Topic | Expected Mode |
|---|---|
| "Self-attention mechanism in Transformers" | `closed_book` |
| "State-of-the-art multimodal LLMs in 2026" | `hybrid` |
| "Latest AI news this week" | `open_book` |
| "How to build a RAG pipeline with LangChain" | `closed_book` or `hybrid` |
