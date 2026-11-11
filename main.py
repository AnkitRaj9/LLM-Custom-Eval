"""
main.py - controls the entire eval flow (ChatOpenRouter version).

For each model:
  1. load the model (via OpenRouter, through ChatOpenRouter)
  2. run the golden dataset (generate SQL for each question)
  3. run the generated SQL on the DB (get a result)
  4. evaluate (compare to gold, using evaluator.py)
  5. log the score

Files it depends on:
  - schema.sql             (the schema, for the prompt)
  - golden_dataset.csv     (questions + gold_sql + order_sensitive)
  - model_openrouter_slug.py  (the 5 models under test -> MODELS)
  - evaluator.py           (the comparison logic)

Setup:
  pip install langchain-openrouter python-dotenv pandas
  .env file with:  OPENROUTER_API_KEY=sk-or-...
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import re
import json
import time
import sqlite3
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from model_openrouter_slug import MODELS
from evaluator import evaluate_one

# ---------- CONFIG ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DB_PATH = "ipl_2021_2024.db"
SCHEMA_PATH = "schema.sql"
GOLDEN_PATH = "golden_dataset.csv"
RESULTS_PATH = "eval_results.csv"

# Cost per 1M tokens: {"model_name": (input_price_per_1M, output_price_per_1M)}
MODEL_PRICING = {
    "gpt-5.6-terra": (2.00, 12.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    # Add fallbacks if testing other models
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
}
# ----------------------------


# ---------- helpers ----------
def load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return f.read().strip()


def load_golden():
    return pd.read_csv(GOLDEN_PATH)


def make_llm(provider, model_name):
    """Instantiate either OpenAI or Gemini based on provider."""
    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=800,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            max_output_tokens=800,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def clean_sql(raw):
    """Strip markdown fences / prose, return runnable SQL."""
    if not raw:
        return ""

    if isinstance(raw, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        )
    else:
        text = str(raw)

    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r"^\s*sql\s*\n", "", text, flags=re.IGNORECASE)
    m = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]
    return text.strip().strip("`").rstrip(";").strip("`").strip()


def extract_tokens(response):
    """Extract input and output token counts across LangChain provider wrappers."""
    input_tokens = 0
    output_tokens = 0

    # 1. Standard usage_metadata
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = response.usage_metadata.get("input_tokens", 0)
        output_tokens = response.usage_metadata.get("output_tokens", 0)
    # 2. Provider response_metadata fallback
    elif hasattr(response, "response_metadata") and response.response_metadata:
        usage = response.response_metadata.get("token_usage", {}) or response.response_metadata.get("usage", {})
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

    return input_tokens, output_tokens


def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculate query cost based on token counts and per-million token rates."""
    pricing = MODEL_PRICING.get(model_name, (0.0, 0.0))
    in_cost = (input_tokens / 1_000_000) * pricing[0]
    out_cost = (output_tokens / 1_000_000) * pricing[1]
    return in_cost + out_cost


def generate_sql(question, schema, llm, model_name, max_retries=3):
    """Ask one model for SQL, tracking latency and token usage."""
    system_msg = (
        "You are a text-to-SQL generator. Given a database schema and a question, "
        "return a single SQL query that answers it. Use SQLite syntax. "
        "Return only the SQL query."
    )
    user_msg = f"Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"

    delay = 2
    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            response = llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ])
            latency = time.perf_counter() - start_time

            input_tokens, output_tokens = extract_tokens(response)
            cost = calculate_cost(model_name, input_tokens, output_tokens)

            raw = response.content
            cleaned = clean_sql(raw)
            if not cleaned:
                print(f"  RAW (uncleaned): {repr(raw)[:300]}")

            return cleaned, latency, input_tokens, output_tokens, cost

        except Exception as e:
            if any(err in str(e).lower() for err in ["503", "unavailable", "429", "rate limit"]):
                if attempt < max_retries - 1:
                    print(f"  [API Transient Error] Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise e


def run_sql(conn, sql):
    """Run SQL on the DB. Returns a DataFrame, or None if it errored."""
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        return None


def gold_result_to_df(gold_result_json):
    """Rebuild the gold result DataFrame from the stored JSON."""
    obj = json.loads(gold_result_json)
    return pd.DataFrame(obj["rows"], columns=obj["columns"])


# ---------- the main flow ----------
def run_eval():
    schema = load_schema()
    golden = load_golden()
    conn = sqlite3.connect(DB_PATH)

    all_rows = []
    scoreboard = {}

    for name, provider, model_name in MODELS:
        print(f"\n{'='*60}\nMODEL: {name}  ({provider} / {model_name})\n{'='*60}")
        llm = make_llm(provider, model_name)
        correct = 0
        total_latency = 0.0
        total_cost = 0.0

        for _, g in golden.iterrows():
            qid = g["id"]
            question = g["question"]
            order_sensitive = str(g["order_sensitive"]).upper() == "TRUE"
            gold_df = gold_result_to_df(g["gold_result_json"])

            # 1. Generate SQL with latency & cost tracking
            try:
                sql, latency, in_tokens, out_tokens, cost = generate_sql(question, schema, llm, model_name)
                total_latency += latency
                total_cost += cost
            except Exception as e:
                print(f"  #{qid:2} [{g['difficulty']:6}] GEN-ERROR: {e}")
                all_rows.append(dict(
                    model=name, id=qid, difficulty=g["difficulty"],
                    correct=False, reason="gen_error", sql="",
                    latency_sec=0.0, input_tokens=0, output_tokens=0, cost_usd=0.0
                ))
                continue

            # 2. Run on DB
            gen_df = run_sql(conn, sql)

            # 3. Evaluate
            verdict = evaluate_one(gold_df, gen_df, order_sensitive)
            if verdict["correct"]:
                correct += 1

            mark = "OK " if verdict["correct"] else "XX "
            print(f"  #{qid:2} [{g['difficulty']:6}] {mark} {verdict['reason']}  ({latency:.2f}s | ${cost:.5f})")

            all_rows.append(dict(
                model=name,
                id=qid,
                difficulty=g["difficulty"],
                correct=verdict["correct"],
                reason=verdict["reason"],
                sql=sql,
                latency_sec=round(latency, 3),
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=round(cost, 6)
            ))

        total = len(golden)
        avg_lat = total_latency / total if total else 0
        scoreboard[name] = {
            "correct": correct,
            "total": total,
            "accuracy": (100 * correct / total) if total else 0,
            "avg_latency": avg_lat,
            "total_cost": total_cost,
        }

        print(f"\n  SCORE: {correct}/{total} = {100*correct/total:.1f}% | Avg Latency: {avg_lat:.2f}s | Total Cost: ${total_cost:.5f}")

    conn.close()

    # 4. Final Scoreboard
    print(f"\n{'='*75}\nFINAL SCOREBOARD (Execution, Latency & Cost)\n{'='*75}")
    print(f"{'Model':<20} | {'Accuracy':<12} | {'Avg Latency':<12} | {'Total Cost':<10}")
    print("-" * 75)
    for name, _, _ in MODELS:
        stats = scoreboard[name]
        acc_str = f"{stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)"
        lat_str = f"{stats['avg_latency']:.2f}s"
        cost_str = f"${stats['total_cost']:.5f}"
        print(f"{name:<20} | {acc_str:<12} | {lat_str:<12} | {cost_str:<10}")

    # 5. Save results to CSV (with lock-safe fallback)
    try:
        pd.DataFrame(all_rows).to_csv(RESULTS_PATH, index=False)
        print(f"\nDetailed results saved -> {RESULTS_PATH}")
    except PermissionError:
        fallback = f"eval_results_{int(time.time())}.csv"
        pd.DataFrame(all_rows).to_csv(fallback, index=False)
        print(f"\nCould not write to {RESULTS_PATH} (open in Excel). Saved -> {fallback}")


if __name__ == "__main__":
    if not OPENAI_API_KEY or not GEMINI_API_KEY:
        print("Missing OPENAI_API_KEY or GEMINI_API_KEY - add them to your .env file")
    else:
        run_eval()