from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, cast

import gradio as gr
from dotenv import load_dotenv

from agents.property_matcher.graph import (
    DEFAULT_PERSONAS,
    PersonaKey,
    PropertyMatcherState,
    run_property_matcher,
)

load_dotenv()

BASE_DIR = Path(__file__).parent
LEADS_PATH = BASE_DIR / "customer_leads.jsonl"
FEEDBACK_PATH = BASE_DIR / "customer_feedback.jsonl"
SUMMARY_PATH = BASE_DIR / "me" / "business_summary.txt"


def read_business_context() -> str:
    """Load the business context presented to the agents."""

    segments: List[str] = []
    if SUMMARY_PATH.exists():
        segments.append(SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        segments.append("NeuraEstate is a Dubai-based AI real estate concierge service.")

    pdf_path = BASE_DIR / "me" / "about_business.pdf"
    if pdf_path.exists():
        segments.append("Additional collateral available in about_business.pdf")

    return "\n\n".join(segments)


BUSINESS_CONTEXT = read_business_context()


def ensure_log_file(path: Path) -> None:
    if not path.exists():
        path.write_text("", encoding="utf-8")


for log_path in (LEADS_PATH, FEEDBACK_PATH):
    ensure_log_file(log_path)


def append_jsonl(path: Path, payload: Dict) -> None:
    ensure_log_file(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def initialise_state() -> PropertyMatcherState:
    return PropertyMatcherState(
        messages=[{"role": "system", "content": BUSINESS_CONTEXT}],
        preferences={},
        next_action="preference_elicitation",
        context=BUSINESS_CONTEXT,
    )


def run_chat_turn(
    user_message: str,
    history: List[Tuple[str, str]] | None,
    agent_selection: Sequence[str],
    temperature: float,
    model_name: str,
    state: PropertyMatcherState | None,
) -> Tuple[List[Tuple[str, str]], PropertyMatcherState, str]:
    if not user_message.strip():
        return history or [], state or initialise_state(), ""

    history = history or []
    state = state or initialise_state()

    state_messages = list(state.get("messages", []))
    state_messages.append({"role": "user", "content": user_message})
    state["messages"] = state_messages
    state["context"] = BUSINESS_CONTEXT

    selected_agents = [cast(PersonaKey, key) for key in agent_selection] if agent_selection else None

    result_state = run_property_matcher(
        state,
        model=model_name.strip() or None,
        temperature=temperature,
        agent_selection=selected_agents,
    )

    assistant_messages = [msg for msg in result_state.get("messages", []) if msg["role"] == "assistant"]
    response_text = assistant_messages[-1]["content"] if assistant_messages else "I'm here to help!"

    history.append((user_message, response_text))

    timestamp = datetime.utcnow().isoformat()
    if lead := result_state.get("lead"):
        lead_payload = {"timestamp": timestamp, **lead}
        append_jsonl(LEADS_PATH, lead_payload)
    if feedback := result_state.get("feedback"):
        feedback_payload = {"timestamp": timestamp, "message": feedback}
        append_jsonl(FEEDBACK_PATH, feedback_payload)

    result_state["context"] = BUSINESS_CONTEXT
    return history, result_state, ""


def reset_chat() -> Tuple[List[Tuple[str, str]], PropertyMatcherState]:
    return [], initialise_state()


def submit_lead(name: str, email: str, phone: str, notes: str) -> str:
    if not (name and (email or phone)):
        return "Please provide at least a name and one contact method."

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": "form",
        "name": name,
        "email": email,
        "phone": phone,
        "notes": notes,
    }
    append_jsonl(LEADS_PATH, payload)
    return "✅ Lead submitted. Our concierge will reach out shortly."


def submit_feedback(message: str) -> str:
    if not message.strip():
        return "Please share a question or feedback message."

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
    }
    append_jsonl(FEEDBACK_PATH, payload)
    return "✅ Thank you! We've logged your feedback."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="NeuraEstate Property Matcher") as demo:
        gr.Markdown(
            """
            # 🏙️ NeuraEstate Property Matcher
            Start a conversation to discover Dubai properties curated by the LangGraph-powered agents.
            """
        )

        conversation_state = gr.State(initialise_state())

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                with gr.Row():
                    agent_selector = gr.CheckboxGroup(
                        choices=[key for key in DEFAULT_PERSONAS],
                        value=[key for key in DEFAULT_PERSONAS],
                        label="Active personas",
                        info="Toggle which agents participate in the LangGraph run.",
                    )
                    with gr.Column():
                        model_name = gr.Textbox(
                            label="Model",
                            value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                            placeholder="gpt-4o-mini",
                        )
                        temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
                user_input = gr.Textbox(
                    label="Your message",
                    placeholder="Describe your ideal property in Dubai...",
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear conversation")
            with gr.Column(scale=1):
                gr.Markdown("## Lead capture")
                lead_name = gr.Textbox(label="Name")
                lead_email = gr.Textbox(label="Email")
                lead_phone = gr.Textbox(label="Phone")
                lead_notes = gr.Textbox(label="Notes", placeholder="Preferred timeline, neighbourhoods, etc.")
                lead_submit = gr.Button("Submit lead")
                lead_status = gr.Markdown()

                gr.Markdown("## Feedback")
                feedback_message = gr.Textbox(label="Question or feedback")
                feedback_submit = gr.Button("Submit feedback")
                feedback_status = gr.Markdown()

        send.click(
            run_chat_turn,
            inputs=[user_input, chatbot, agent_selector, temperature, model_name, conversation_state],
            outputs=[chatbot, conversation_state, user_input],
        )
        user_input.submit(
            run_chat_turn,
            inputs=[user_input, chatbot, agent_selector, temperature, model_name, conversation_state],
            outputs=[chatbot, conversation_state, user_input],
        )

        clear.click(reset_chat, outputs=[chatbot, conversation_state])

        lead_submit.click(
            submit_lead,
            inputs=[lead_name, lead_email, lead_phone, lead_notes],
            outputs=lead_status,
        )
        feedback_submit.click(
            submit_feedback,
            inputs=[feedback_message],
            outputs=feedback_status,
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue(max_size=32).launch()


if __name__ == "__main__":
    main()
