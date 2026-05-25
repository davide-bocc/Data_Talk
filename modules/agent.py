"""
agent.py
--------------------
Natural-language Q&A agent powered by Google Gemini 1.5 Flash (free tier).

Key design decisions:
  - Uses the user's GEMINI_API_KEY from .env (local) or Streamlit Secrets (cloud)
  - Sends the actual CSV data to the model, not just a summary
  - Adapts how much data it sends based on file size (adaptive strategy)
  - Enforces a per-session question limit to protect the free API quota

Adaptive data strategy:
  < 5,000 rows -> full CSV as text
  5,000–50,000 -> stratified sample + full statistics
  > 50,000 -> statistics + small sample + explicit warning

Usage:
    from modules.agent import build_context, ask, check_limit, QUESTION_LIMIT

    context  = build_context(dataset, report)
    allowed, used, remaining = check_limit(st.session_state)
    if allowed:
        response = ask(question, context, history, st.session_state)
        st.markdown(response)
    else:
        st.warning(f"Session limit of {QUESTION_LIMIT} questions reached.")
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from modules.loader   import LoadedDataset, SEMANTIC_TYPES
from modules.analyzer import AnalysisReport

# Load .env file when running locally.
# On Streamlit Cloud this has no effect - secrets are already in the environment.
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

#  Configuration

QUESTION_LIMIT = 20          # max questions per browser session
MAX_ROWS_FULL = 5_000       # below this → send full CSV
MAX_ROWS_SAMPLE = 50_000      # below this → send stratified sample
SAMPLE_SIZE = 300         # rows to sample when file is large
MODEL_NAME = "gemini-2.5-flash"


#  1. Data structures
@dataclass
class DatasetContext:

    # Everything the model needs to answer questions about the dataset.
    # The 'data_text' field contains the actual CSV rows (full or sampled).
    # The 'stats_text' field always contains complete statistics.

    meta_text: str    # filename, shape, memory
    schema_text: str    # column names, types, missing %
    data_text: str    # actual data rows (CSV format)
    stats_text: str    # descriptive statistics from analyzer.py
    missing_text: str    # missing value table
    insights_text: str    # auto-generated insights
    data_mode: str    # "full" | "sample" | "stats_only"
    n_rows_sent: int    # how many rows were actually sent


@dataclass
class AgentMessage:
    # A single turn in the conversation.
    role: str    # Gemini uses "model", not "assistant"
    content: str


@dataclass
class ConversationHistory:

    # Stores the full conversation so each API call has memory of previous turns.
    # Gemini requires role alternation: user -> model -> user -> model ...
    # The first message must always be from the user.

    messages: list[AgentMessage] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(AgentMessage(role=role, content=content))

    def to_api_format(self) -> list[dict]:
        """Converts history to the format expected by the Gemini SDK."""
        return [
            {"role": m.role, "parts": [{"text": m.content}]}
            for m in self.messages
        ]

    def clear(self) -> None:
        self.messages.clear()


#  2. Rate limiter
_COUNT_KEY = "datatalk_question_count"


def check_limit(session_state: dict) -> tuple[bool, int, int]:
    """
    Reads the question counter from Streamlit's session_state.

    Returns
    -------
    allowed: True if the user can still ask questions
    used: how many questions have been asked this session
    remaining: how many questions are left
    """
    used = session_state.get(_COUNT_KEY, 0)
    remaining = max(0, QUESTION_LIMIT - used)
    return remaining > 0, used, remaining


def _increment_counter(session_state: dict) -> None:
    # Increments the question counter in session_state
    session_state[_COUNT_KEY] = session_state.get(_COUNT_KEY, 0) + 1

#  3. Adaptive data builder
def _build_data_text(df: pd.DataFrame) -> tuple[str, str, int]:
    """
    Decides how much of the DataFrame to send to the model based on row count.

    Returns
    -------
    data_text  : the actual data as a CSV string
    data_mode  : "full" | "sample" | "stats_only"
    n_rows_sent: number of rows included in data_text
    """
    n = len(df)

    if n <= MAX_ROWS_FULL:
        # Small file — send everything
        data_text = df.to_csv(index=False)
        data_mode = "full"
        n_rows_sent = n

    elif n <= MAX_ROWS_SAMPLE:
        # Medium file — stratified sample:
        # take the first 100, last 100, and 100 random rows from the middle
        chunk = SAMPLE_SIZE // 3
        head = df.head(chunk)
        tail = df.tail(chunk)
        mid = df.iloc[chunk: n - chunk].sample(
            n=min(chunk, n - 2 * chunk), random_state=42
        )
        sampled = pd.concat([head, mid, tail]).drop_duplicates()
        data_text = sampled.to_csv(index=False)
        data_mode = "sample"
        n_rows_sent = len(sampled)

    else:
        # Large file — send only a small representative sample
        sampled = df.sample(n=min(SAMPLE_SIZE, n), random_state=42)
        data_text = sampled.to_csv(index=False)
        data_mode = "stats_only"   # name reflects intent: stats are primary
        n_rows_sent = len(sampled)

    return data_text, data_mode, n_rows_sent

#  4. Context builder
def build_context(
    dataset: LoadedDataset,
    report:  AnalysisReport,
) -> DatasetContext:
    """
    Converts a LoadedDataset + AnalysisReport into a DatasetContext
    that will be injected into every API call.

    This is called once when the user uploads a file, not on every question.
    The result is stored in Streamlit's session_state so it is not rebuilt
    on every Streamlit rerun.
    """
    df = dataset.df

    # -- Meta --
    m = dataset.meta
    meta_text = (
        f"Filename : {m['filename']}\n"
        f"Rows : {m['n_rows']:,}\n"
        f"Columns : {m['n_cols']}\n"
        f"Memory : {m['memory_kb']} KB\n"
        f"Missing : {m['pct_missing']}% of all cells"
    )

    # -- Schema --
    lines = ["Column | Type | Semantic | Unique | Missing"]
    lines.append("-" * 90)
    for col in dataset.columns:
        lines.append(
            f"{col.name:<30} | {col.dtype_raw:<12} | "
            f"{SEMANTIC_TYPES.get(col.semantic, col.semantic):<20} | "
            f"{col.n_unique:<6} | {col.pct_missing}%"
        )
    schema_text = "\n".join(lines)

    # -- Actual data (adaptive) --
    data_text, data_mode, n_rows_sent = _build_data_text(df)

    # -- Statistics --
    stats_text = (
        report.summary_df.to_string()
        if not report.summary_df.empty
        else "No numeric columns available."
    )

    # -- Missing values --
    missing_text = (
        report.missing_df.to_string()
        if not report.missing_df.empty
        else "No missing values detected."
    )

    # -- Insights --
    insights_text = "\n".join(f"- {i}" for i in report.insights)

    return DatasetContext(
        meta_text=meta_text,
        schema_text=schema_text,
        data_text=data_text,
        stats_text=stats_text,
        missing_text=missing_text,
        insights_text=insights_text,
        data_mode=data_mode,
        n_rows_sent=n_rows_sent,
    )


def _context_to_prompt(context: DatasetContext) -> str:

    # Assembles the DatasetContext into the opening message sent to Gemini.
    # Includes a note about the data mode so the model knows if it has
    # partial or complete data.

    data_note = {
        "full": f"The COMPLETE dataset ({context.n_rows_sent:,} rows) is provided below.",
        "sample": f"A STRATIFIED SAMPLE of {context.n_rows_sent:,} rows is provided below "
                  f"(the full dataset is larger). Statistics are computed on the full data.",
        "stats_only": f"The dataset is very large. A RANDOM SAMPLE of {context.n_rows_sent:,} rows "
                      f"is provided. Statistics are computed on the full data. "
                      f"Mention this limitation when relevant.",
    }.get(context.data_mode, "")

    return textwrap.dedent(f"""
        You are analysing a dataset uploaded by the user.
        {data_note}

        ## Dataset Overview
        {context.meta_text}

        ## Schema
        {context.schema_text}

        ## Data
        {context.data_text}

        ## Descriptive Statistics (computed on full dataset)
        {context.stats_text}

        ## Missing Values
        {context.missing_text}

        ## Key Insights (auto-generated)
        {context.insights_text}
    """).strip()

#  5. System prompt
_SYSTEM_PROMPT = textwrap.dedent("""
    You are Data_Talk, an expert data analyst assistant embedded in a web app.
    You help business users understand their data through clear, jargon-free answers.

    Rules:
    - Answer ONLY based on the data and statistics provided. Never invent numbers.
    - If you cannot answer from the available data, say so honestly.
    - Format numbers with thousands separators and 2 decimal places where appropriate.
    - Keep answers concise but complete. Use bullet points for lists.
    - If you notice an important insight the user did not ask about, briefly mention it.
    - Always respond in the same language the user writes in.
    - If the dataset is sampled, acknowledge this limitation when the answer
      might differ on the full data.
""").strip()

#  6. Main ask function

def ask(
    question: str,
    context: DatasetContext,
    history: ConversationHistory,
    session_state: dict,
    api_key: Optional[str] = None,
) -> str:
    """
    Sends the user's question to Gemini 1.5 Flash and returns the answer.

    On the first question the full dataset context is injected into the
    conversation. On subsequent questions only the new question is sent —
    Gemini remembers the context through the conversation history.

    Parameters
    ----------
    question: the user's natural-language question
    context: DatasetContext built by build_context()
    history: ConversationHistory — updated in place after each call
    session_state: Streamlit's st.session_state — used for rate limiting
    api_key: optional override; falls back to GEMINI_API_KEY env var
    """
    # -- Rate limit check --
    allowed, used, remaining = check_limit(session_state)
    if not allowed:
        return (
            f"You have reached the session limit of **{QUESTION_LIMIT} questions**. "
            f"Reload the page to start a new session."
        )

    # -- Resolve API key --
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return (
            "Gemini API key not found. "
            "Set **GEMINI_API_KEY** in your `.env` file (local) "
            "or in Streamlit Cloud Secrets (deployed app)."
        )

    # -- Import Gemini SDK (lazy) --
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return (
            "The `google-genai` package is not installed. "
            "Run: pip install google-genai"
        )

    client = genai.Client(api_key=key)

    # -- Build message list --
    is_first_turn = len(history.messages) == 0

    if is_first_turn:
        # First turn: include the full dataset context in the opening message
        opening = _context_to_prompt(context) + "\n\n---\n\nFirst question: " + question
        api_messages = [{"role": "user", "parts": [{"text": opening}]}]
    else:
        # Subsequent turns: context is already in history
        api_messages = history.to_api_format() + [
            {"role": "user", "parts": [{"text": question}]}
        ]

    # -- Call the API --
    try:
        contents = []
        for msg in api_messages:
            contents.append(
                types.Content(
                    role=msg["role"],
                    parts=[types.Part(text=msg["parts"][0]["text"])]
                )
            )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=1024,
            )
        )
        answer = response.text

        # -- Update history --
        if is_first_turn:
            opening_msg = _context_to_prompt(context) + "\n\n---\n\nFirst question: " + question
            history.add(role="user",  content=opening_msg)
        else:
            history.add(role="user",  content=question)
        history.add(role="model", content=answer)

        # -- Increment rate limit counter --
        _increment_counter(session_state)

        return answer



    except Exception as exc:
        error = str(exc).lower()
        if "api_key" in error or "authentication" in error or "403" in error:
            return "Invalid API key. Check your GEMINI_API_KEY."
        if "quota" in error or "rate" in error or "429" in error:
            return "Troppe richieste ravvicinate. Aspetta 1 minuto e riprova."
        if "candidate" in error or "safety" in error:
            return "Gemini ha bloccato questa risposta. Prova a riformulare."
        if "404" in error or "not found" in error:
            return "Modello non trovato. Controlla MODEL_NAME in agent.py."
        return f"Errore: {str(exc)[:200]}"

#  7. Suggested starter questions

def suggest_questions(dataset: LoadedDataset, report: AnalysisReport) -> list[str]:

    # Returns up to 8 relevant starter questions based on the actual
    # content of the dataset. Shown in Streamlit as clickable buttons.

    questions: list[str] = [
        "Give me a general overview of this dataset.",
        "Are there any data quality issues I should know about?",
    ]

    if dataset.numeric_cols:
        col = dataset.numeric_cols[0]
        questions.append(f"What is the distribution of '{col}'?")
        questions.append(f"Are there outliers in '{col}'?")

    if dataset.categorical_cols:
        col = dataset.categorical_cols[0]
        questions.append(f"What are the most common values in '{col}'?")

    if len(dataset.numeric_cols) >= 2:
        a, b = dataset.numeric_cols[0], dataset.numeric_cols[1]
        questions.append(f"Is there a correlation between '{a}' and '{b}'?")

    if dataset.datetime_cols and dataset.numeric_cols:
        v = dataset.numeric_cols[0]
        questions.append(f"How does '{v}' trend over time?")

    if not report.anomalies_df.empty:
        questions.append("Tell me about the anomalies found in the data.")

    return questions[:8]