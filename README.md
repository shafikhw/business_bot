# NeuraEstate AI Business Assistant

A production-ready AI-powered business assistant for **NeuraEstate**, a UAE-based AI real estate technology company. Built with Gradio for deployment on Hugging Face Spaces.

## Features

- 🤖 AI-powered chatbot that answers questions about NeuraEstate
- 📝 Automatic lead capture when users express interest
- 💬 Feedback collection for unanswered questions
- 📊 Health monitoring and chat history export
- 🎨 Modern, user-friendly Gradio interface

## Quick Start

### Local Development

1. **Activate your environment:**
   ```bash
   source C:/Users/********/anaconda3/Scripts/activate ai-agentic
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your-api-key-here
   PROVIDER=openai
   ```

4. **Run the app:**
   ```bash
   python app.py
   ```

5. **Access the interface:**
   
   Open your browser to `http://localhost:7860`

### Hugging Face Spaces Deployment

1. **Create a new Space:**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Create a new Space
   - Select "Gradio" as the SDK
   - Set visibility (Public/Private)

2. **Clone and push your repository:**
   ```bash
   git clone <your-space-repo-url>
   cd <your-space-repo>
   # Copy your files (app.py, requirements.txt, runtime.txt, me/ folder)
   git add .
   git commit -m "Initial commit"
   git push
   ```

3. **Set Repository Secrets:**
   
   In your Space settings:
   - Go to **Settings** → **Repository secrets**
   - Add secret: `OPENAI_API_KEY` with your API key value
   - Optionally add: `PROVIDER=openai`

4. **Build and Deploy:**
   
   The Space will automatically build when you push. Monitor the build logs in the Space interface.

## Project Structure

```
business_bot/
├── .env                    # Local environment variables (not committed)
├── app.py                  # Main Gradio application
├── requirements.txt        # Python dependencies
├── runtime.txt            # Python version specification
├── README.md              # This file
├── me/
│   ├── about_business.pdf # Business information (PDF)
│   └── business_summary.txt # Business summary (text)
└── logs/                  # Auto-created directory
    ├── leads.jsonl        # Customer leads (auto-generated)
    └── feedback.jsonl     # Customer feedback (auto-generated)
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `PROVIDER` (optional): LLM provider, defaults to "openai"

### Files Required

The app expects these files in the `me/` directory:
- `business_summary.txt`: Text summary of NeuraEstate
- `about_business.pdf`: Additional business information

## Usage

### Chat Interface

1. Type your questions in the message box
2. The assistant will answer based on the loaded business context
3. If you express interest (mention buying, selling, contact info), your information will be automatically captured

### Lead Capture Form

Use the sidebar form to directly submit your contact information:
1. Fill in your Name and Email
2. Add any notes about your interest
3. Click "Submit Lead"

### Export Chat History

Click the "📥 Export Chat History" button to download your conversation as JSON.

### Health Check

The sidebar shows system health status. Click "🔄 Refresh Health" to update.

## Logs

The app automatically creates a `logs/` directory and writes:
- `leads.jsonl`: Customer leads (one JSON object per line)
- `feedback.jsonl`: Unanswered questions/feedback (one JSON object per line)

Each entry includes:
- `timestamp`: ISO 8601 UTC timestamp
- Relevant fields (name, email, message, question)

## Troubleshooting

### "API Key Missing" Warning

**Local:**
- Ensure `.env` file exists with `OPENAI_API_KEY=your-key`
- Check that `python-dotenv` is installed

**Hugging Face Spaces:**
- Go to Space Settings → Repository secrets
- Add `OPENAI_API_KEY` secret with your key value
- Rebuild the Space

### Business Context Not Loading

- Ensure `me/business_summary.txt` and `me/about_business.pdf` exist
- Check file permissions (readable)
- Review console logs for specific error messages

### Build Failures on Spaces

- Check `runtime.txt` specifies a valid Python version (3.10+)
- Verify all dependencies in `requirements.txt` are available
- Review build logs in the Spaces interface

## Development

### Code Structure

- **LLM Adapter**: Thin abstraction layer for LLM providers (`LLM` class)
- **Tool Functions**: `record_customer_interest()`, `record_feedback()`
- **Heuristics**: Automatic detection of lead intent and uncertainty
- **Gradio UI**: Blocks-based interface with sidebar and main chat

### Adding New Providers

Extend the `LLM` class in `app.py`:

```python
if self.provider == "your_provider":
    # Initialize your provider client
    # Implement chat() method
```

## License

This project is for educational/demo purposes.

## Contact

For questions about NeuraEstate, contact through the chatbot interface.

