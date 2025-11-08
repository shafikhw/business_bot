# NeuraEstate Property Matcher

A LangGraph-powered AI assistant that helps NeuraEstate's clients discover Dubai properties, capture qualified leads, and collect product feedback. The system ships with a Gradio front end, orchestrated multi-agent workflow, and JSONL logging so you can run the full experience locally.

## Features

- **Multi-agent conversation** – Three specialised personas (preference specialist, Bayut scout, concierge) coordinate via LangGraph to collect requirements and recommend listings.
- **Lead + feedback capture** – Conversations are monitored for buying intent and unanswered questions. Structured events are persisted to JSONL logs alongside manual form submissions.
- **Context-aware answers** – Business collateral from the `me/` directory is loaded on startup and injected into every LangGraph run.
- **Production ready UI** – A modern Gradio Blocks interface with chat history, agent toggles, lead forms, and health indicators.

## Requirements

- Python 3.10 or newer (tested on Python 3.11)
- An OpenAI API key with access to the `gpt-4o-mini` (or compatible) model
- (Optional) Mapbox access token if you want property enrichment that depends on the `MAPS_PROVIDER`

## 1. Clone the repository

```bash
git clone https://github.com/<your-org>/business_bot.git
cd business_bot
```

## 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows (PowerShell): .venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini      # optional override, defaults to gpt-4o-mini
PROVIDER=openai               # optional, currently only openai is implemented
MAPS_PROVIDER=mapbox          # optional, enables map enrichment
MAPBOX_ACCESS_TOKEN=pk.your-token  # required when MAPS_PROVIDER=mapbox
```

The application reads this file automatically through `python-dotenv` when `app.py` starts. If you skip the Mapbox configuration the assistant still runs, but map-dependent features will emit a polite warning.

## 5. Verify required collateral

Ensure the following business context files exist:

- `me/business_summary.txt`
- `me/about_business.pdf`

The included samples are loaded automatically. Replace them with your own collateral as needed.

## 6. Run the application locally

```bash
python app.py
```

Gradio launches on `http://127.0.0.1:7860` by default. Open the URL in your browser to start chatting. The interface supports:

- Sending free-form messages to the LangGraph workflow
- Toggling which personas are active (useful for demos or troubleshooting)
- Submitting leads through the sidebar form
- Logging unresolved questions through the feedback form

## 7. Review generated data

Two JSONL files are maintained in the project root:

- `customer_leads.jsonl` – automatic and manual leads with timestamps and metadata
- `customer_feedback.jsonl` – unresolved questions or feedback strings

Each run appends, so consider rotating these files between sessions if you want a clean slate.

## 8. Run automated tests

The repository ships with pytest coverage for the Bayut tool chain. With your virtual environment activated:

```bash
pytest
```

All tests should pass (expect output similar to `2 passed`).

## 9. Troubleshooting

- **`OPENAI_API_KEY` errors** – Confirm the key exists in `.env` and that the model specified in `OPENAI_MODEL` is available to your account.
- **Port already in use** – Pass a different port to Gradio, e.g. edit `demo.queue(...).launch(server_port=7861)` inside `app.py`.
- **Offline fallback** – If the OpenAI client errors, each persona emits a heuristic response so the UI stays responsive. Fix your network/API key to regain full-quality answers.

## 10. Optional: Deploy to Hugging Face Spaces

The app runs unmodified on Hugging Face Spaces using the `gradio` SDK. Create a new Space, push the repository contents, and add the same environment variables as repository secrets.

## Project structure

```
business_bot/
├── app.py                     # Gradio entrypoint
├── agents/                    # LangGraph personas and tooling
├── customer_leads.jsonl       # Lead log (auto-created)
├── customer_feedback.jsonl    # Feedback log (auto-created)
├── me/                        # Business collateral loaded into the assistant
├── requirements.txt           # Python dependencies
├── tests/                     # Pytest suite
└── README.md                  # This document
```

## Support

For issues specific to NeuraEstate operations, reach out via the chatbot or your internal NeuraEstate support channel. For bugs in this repository, open a GitHub issue with logs and reproduction steps.
