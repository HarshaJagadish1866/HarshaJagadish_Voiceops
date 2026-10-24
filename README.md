# ZentroMart AI Customer Support Agent

An agentic, multimodal AI customer support solution built for **ZentroMart**, a fictional Indian e-commerce marketplace. The agent handles multi-turn support interactions across voice and text, strictly grounded in company policy documents, with relational database lookups, business day calendar calculations, identity verification gatekeeping, and speech capabilities powered by the Gnani Speech AI platform.

---

## Architecture Overview

```text
Incoming Audio (.wav)
       │
       ▼
[ Gnani STT (en-IN) ]  ──> Customer Transcript
                                │
                                ▼
                   [ ZentroMart Conversation Agent ]
                   ├── Identity Verification Gatekeeper (Phone suffix / Pincode)
                   ├── Relational DB Lookups (Orders, Items, Customers, Pincodes)
                   ├── Business Day Calculator (Mon-Sat, skipping holidays.csv)
                   ├── RAG System (21 KB docs, Precedence Ranking)
                   └── Policy Safeguards & Mock Tools (lookup_order, create_ticket)
                                │
                                ▼
                       Agent Text Response
                                │
                                ▼
              [ Natural Speech Text Normalization ]
              (Spaced order IDs, spoken years 'twenty twenty-six', rupees)
                                │
                                ▼
         [ Gnani TTS (timbre-v2.5, Kaveri, en-IN) ]
                                │
                                ▼
                     Spoken Audio Reply (.wav)
```

---

## Deliverables & Project Structure

- **`run.py`**: Master execution script. Takes a folder of customer audio files, runs the full pipeline (`STT -> RAG/Tools -> TTS`), and outputs the text reply, the spoken `.wav` audio reply, and the retrieved document chunks used.
- **`run.sh`**: Executable bash script wrapper for `run.py`.
- **`load_database.py`**: Loads the 7 CSV files with `dtype=str`, maps relational joins, sets the reference date (20 October 2026), and provides business day calculation logic.
- **`rag_system.py`**: Chunks all 21 Markdown files in `kb/`, preserving `[SECTION-ID]`, `status`, and `effective_from`. Enforces document precedence (`Current Policy -> Agent Guide -> Help -> FAQ -> Marketing`), and runs vector retrieval with strict grounding.
- **`conversation_agent.py`**: Upgrades the agent to handle multi-turn conversations, state memory, identity verification gatekeeping, policy refusals, and human handover ticket creation.
- **`voice_agent.py`**: Integrates Gnani STT (`en-IN`) and Gnani TTS (`timbre-v2.5`, `Kaveri`, `en-IN`) with speech text normalization.
- **`.env.example`**: Clean environment variable template without real secrets.
- **`requirements.txt`**: Python dependencies.

---

## Quick Start Guide

### 1. Prerequisites
- **Python**: 3.10+ (tested on Python 3.13)
- An active Gnani API Key (obtain from [vachana.ai](https://vachana.ai))

### 2. Installation

Clone or extract the repository, then install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and set your Gnani API key:

```bash
cp .env.example .env
```

Open `.env` and configure:
```ini
GNANI_API_KEY="your_gnani_vachana_api_key_here"
```

*(Note: API keys are loaded dynamically via `python-dotenv` and are never hardcoded).*

---

## Running the Master Pipeline

To run the full end-to-end pipeline across a directory of incoming customer audio recordings:

### Using Python:
```bash
python3 run.py <path_to_audio_folder> --output-dir <path_to_output_folder>
```

**Example:**
```bash
python3 run.py sample_incoming_audio/ --output-dir output_replies/
```

### Using the Shell Wrapper:
```bash
./run.sh sample_incoming_audio/ output_replies/
```

*(If no directory is specified, `run.py` will automatically synthesize a sample customer audio file and demonstrate the complete loop).*

### What the Master Script Outputs
For every customer audio file processed:
1. **Transcribed Speech**: Customer speech extracted via Gnani STT.
2. **Retrieved Document Chunks**: Exact `[SECTION-ID]`, document title, status, and relevance score used to formulate the policy answer.
3. **Text Reply**: The grounded, polite agent answer adhering to identity verification and policy rules.
4. **Spoken Audio Reply**: Spoken `.wav` audio synthesized with Gnani TTS (`Kaveri`, `timbre-v2.5`).
5. **Audit Report**: A comprehensive JSON file saved to `<output_dir>/results_summary.json`.

---

## Detailed Component Documentation

### 1. Database & Business Days (`load_database.py`)
- Reads all 7 data tables as strings (`dtype=str`):
  - `customers.csv` (300 rows)
  - `orders.csv` (1,000 rows)
  - `order_items.csv` (1,783 rows)
  - `products.csv` (70 rows)
  - `pincodes.csv` (36 rows)
  - `holidays.csv` (4 rows)
  - `support_logs.csv` (244 rows)
- **Relational Joins**:
  - `orders` $\leftrightarrow$ `customers` on `customer_id`
  - `order_items` $\leftrightarrow$ `orders` on `order_id`
  - `order_items` $\leftrightarrow$ `products` on `sku`
  - `orders.delivery_pincode` $\leftrightarrow$ `pincodes.pincode`
- **Business Day Rules**:
  - Reference anchor: **20 October 2026** (`TODAY`).
  - Business days run **Monday through Saturday** (Sundays are non-working).
  - Skips warehouse closure/festival holidays from `data/holidays.csv`.

### 2. Knowledge Base & RAG System (`rag_system.py`)
- **84 Sections Chunked**: Parses all 21 Markdown files in `kb/`. Every chunk preserves `section_id`, `status` (`current` vs `superseded`), `effective_from`, and `doc_type`.
- **Precedence Hierarchy**:
  $$\text{Current Policy} \longrightarrow \text{Agent Guide} \longrightarrow \text{Help} \longrightarrow \text{FAQ} \longrightarrow \text{Marketing} \longrightarrow \text{Superseded (reference only)}$$
- **Hybrid Vector Index**: Sublinear TF-IDF combining word n-grams $(1, 2)$ and character n-grams $(3, 5)$ with catalog product enrichment.
- **Strict Grounding**: Answers strictly from retrieved facts and outputs an explicit "does not know" fallback for unsupported topics.

### 3. Multi-Turn Conversational Agent (`conversation_agent.py`)
- **Self-Introduction ([STY-1])**: Always introduces as ZentroMart's AI support assistant on Turn 1.
- **Identity Verification Gatekeeper ([IDV-1, IDV-2, IDV-3])**:
  - Order details are withheld until the caller provides the **last 4 digits of their registered phone** or **delivery pincode**.
  - Maximum 1 retry allowed on failed verification. On the second failure, zero order details are disclosed.
- **Policy Refusal**:
  - Refuses cancellation of already shipped orders ([CAN-2]).
  - Refuses compensation/vouchers on late Standard deliveries ([DEL-4]).
  - Refuses returns for non-returnable categories ([RET-5]).
- **Human Escalation ([ESC-1])**:
  - Handover triggers: High-value damage ($> \text{₹}10,000$), repeat contacts $\ge 2$, customer requesting a human, overdue refunds.
  - Generates detailed summary ($\ge 20$ chars) and calls `create_ticket`.
  - **Only confirms handover to customer if `created=True`**.
- **Benchmark Performance**: Evaluated against [practice/conversations.csv](file:///Users/harsha_jagadish/Desktop/Capstone_Project_Dataset_FTTT_2026/practice/conversations.csv) across all **96 turns**, achieving a **100.0% pass rate**.

### 4. Speech AI Pipeline (`voice_agent.py`)
- **STT**: `GnaniSTTClient(api_key=...)` with `language_code="en-IN"`.
- **Speech Text Normalization**:
  - Spaces out 8-digit order IDs: `89782837` $\rightarrow$ `"8 9 7 8, 2 8 3 7"`.
  - Speaks years naturally: `2026` $\rightarrow$ `"twenty twenty-six"`.
  - Expands dates: `2026-10-24` $\rightarrow$ `"the 24th of October, twenty twenty-six"`.
  - Converts currency: `₹49` $\rightarrow$ `"49 rupees"`.
  - Strips markdown formatting and raw bracket IDs.
- **TTS**: `GnaniTTSClient(api_key=...)` with `model="timbre-v2.5"`, `voice="Kaveri"`, and `language="en-IN"`, producing 44.1 kHz `.wav` replies.

---

## Running Individual Tests

- **Test Database & Calendar**:
  ```bash
  python3 load_database.py
  ```
- **Test Knowledge Base RAG Search**:
  ```bash
  python3 rag_system.py
  ```
- **Test Multi-Turn Conversational Agent**:
  ```bash
  python3 conversation_agent.py
  ```
- **Test Voice Pipeline**:
  ```bash
  python3 voice_agent.py
  ```
- **Run Benchmark Evaluation (Prompt-Only vs. RAG)**:
  ```bash
  python3 evaluate_rag_comparison.py
  ```

---

## Evaluation Benchmark: Prompt-Only Baseline vs. RAG System

The benchmark script `evaluate_rag_comparison.py` evaluates 52 benchmark questions across 14 intent categories comparing a zero-shot prompt-only LLM against our precedence-grounded RAG system:

| Metric | Prompt-Only (No RAG) | With RAG Enabled | Improvement |
| :--- | :---: | :---: | :---: |
| **Questions Evaluated** | 52 | 52 | — |
| **Passed Questions** | 10 | **52** | **+42 questions** |
| **Overall Accuracy** | **19.23%** | **100.00%** | **+80.77 percentage points** |
| **Relative Performance** | 1.00x | **5.20x** | **520% of baseline** |
| **Hallucination Rate** | **80.77%** | **0.00%** | **Eliminated (0%)** |
| **Grounding Citation Provenance** | 0.0% | **100.0%** | **Verified `[SECTION-ID]` on all replies** |

Exported audit files:
- `rag_vs_prompt_evaluation.csv`
- `rag_vs_prompt_evaluation.json`

---

## Repository Structure

```text
HarshaJagadish_Voiceops/
├── README.md                  # Comprehensive instructions, setup, and evaluation
├── .env.example               # Clean environment template (no secrets)
├── requirements.txt           # Python package dependencies
├── run.py                     # Master batch execution script
├── run.sh                     # Master shell script wrapper
└── src/                       # Core Python implementation modules
    ├── conversation_agent.py  # Multi-turn agent, state memory, identity gatekeeper
    ├── evaluate_rag_comparison.py # Evaluation script comparing Prompt-Only vs RAG
    ├── load_database.py       # Relational CSV loader & business holiday calendar
    ├── rag_system.py          # Vector RAG, section chunking, precedence ranking
    └── voice_agent.py         # Gnani STT & TTS speech normalization pipeline
```

---

## Public Repository
- **GitHub**: [https://github.com/HarshaJagadish1866/HarshaJagadish_Voiceops](https://github.com/HarshaJagadish1866/HarshaJagadish_Voiceops)

