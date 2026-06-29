# Physics Exam OCR and AI Feedback Pipeline (Portfolio + Technical README)

This repository showcases an end-to-end AI pipeline I built for Thai physics exam analysis. It combines OCR, question-level classification, and LLM-based feedback generation to produce structured, student-specific insights and printable reports.

For hiring teams, this project demonstrates practical skills in applied AI engineering: workflow orchestration (LangGraph), data modeling, prompt-driven analysis, and production-style output generation (JSON + PDF) with token-usage observability.

## Portfolio Purpose (For Job Applications)

### Problem I Solved
Teachers and tutoring teams often spend significant time reviewing scanned exam papers, extracting question content, and writing actionable student feedback. This project automates that workflow from document ingestion to final report generation.

### My Contribution
- Designed and implemented a multi-stage AI pipeline for OCR + feedback generation.
- Structured the system using graph-based execution modes (sequential/manual/parallel).
- Built reusable data schemas for reliable state passing across pipeline stages.
- Added reporting outputs and token analytics to support quality review and cost visibility.

### Impact and Value
- Reduces manual effort in turning raw exam files into student feedback artifacts.
- Standardizes feedback output in machine-readable and human-readable formats.
- Enables experimentation across model backends and execution strategies.

### Skills Demonstrated
- Python system design for AI workflows
- OCR processing and document-to-structured-data conversion
- LLM integration (Gemini, Ollama, OpenAI-compatible endpoints)
- LangGraph orchestration and pipeline state management
- Pydantic/TypedDict schema design
- Reporting automation (HTML/PDF with Playwright + MathJax)
- Token usage tracking and analysis dashboarding

## Technical Overview

## What It Does

- Reads exam PDFs and converts pages to images or base64 payloads for processing.
- Runs OCR and question analysis on each page.
- Collects student information and answer-key data from CSV files.
- Produces AI-generated feedback for each student.
- Exports OCR and feedback results to `files_log/`.
- Generates a styled PDF report from the final feedback JSON.
- Includes a token-usage dashboard for reviewing model consumption.

## Main Workflow

1. Load the exam PDF, student answer data, and label definitions.
2. Run the OCR stage to extract question content and classify each item.
3. Read student information and score data.
4. Generate feedback with the selected LLM backend.
5. Save the final outputs as JSON and optionally render them to PDF.

## Project Structure

- `app.py` - main pipeline entry point.
- `Node/` - OCR, feedback, manual processing, and PDF helpers.
- `graphs/` - LangGraph orchestration for sequential, manual, and parallel flows.
- `Schemes/` - Pydantic and TypedDict data models used across the pipeline.
- `Documents/` - curriculum data and CSV inputs.
- `files_log/` - generated OCR and feedback results.
- `history_log/` - historical run outputs and archived logs.
- `report.py` - converts feedback JSON into an HTML/PDF report.
- `token_analytics_dashboard.py` - dashboard for token usage and log review.

## Requirements

- Python 3.10 or newer.
- A virtual environment.
- One or more model backends depending on your chosen run mode:
  - Gemini API
  - Ollama local models
  - Typhoon / OpenAI-compatible endpoint

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with the API keys you plan to use:

```bash
gemini_api_key=your_gemini_api_key
typhoon_api_key=your_typhoon_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
```

## Usage

The default entry point is `app.py`. Before running, update the paths in the script so they match your local files:

- PDF file path
- student answer CSV
- label CSV

Then run:

```bash
python app.py
```

The script writes results to:

- `files_log/final_ocr_results_M5.json`
- `files_log/final_feedback_results_M5.json`

## PDF Report Generation

If you already have feedback JSON and want a formatted report, run:

```bash
python report.py
```

This generates a PDF report in `files_log/` using Playwright and MathJax.

## Model Modes

The pipeline supports different execution styles through the graph builders in `graphs/`:

- `sequential` - process pages step by step.
- `manual` - use a manual OCR stage for page handling.
- `parallel` - process pages concurrently with graph-based orchestration.

Model selection is configured in `config.py`.

## Notes

- The repository contains Thai-language prompts and labels because it targets Thai physics exam workflows.
- Some paths in `app.py` are hardcoded for local experimentation and should be adjusted for your own dataset.
- The project produces both analysis outputs and historical logs, which can be useful for benchmarking model quality.

## Recruiter Quick Summary

If you are evaluating this project for a role, key highlights are:
- End-to-end ownership of an applied AI workflow.
- Practical integration of OCR + LLM components.
- Engineering focus on both output quality and operational visibility.
- Real-world, domain-specific use case (education assessment feedback).

## License

No license file is included in the repository. Add one if you plan to distribute or reuse the project publicly.
