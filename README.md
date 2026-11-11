# LLM-Custom-Eval
# LLM Text-to-SQL Benchmark 🚀

An automated benchmarking pipeline comparing **GPT-5.6 Terra** and **Gemini Flash Lite** on real-world Text-to-SQL generation tasks using actual SQLite database execution.

The project evaluates LLMs based on execution accuracy, query generation latency, and API cost rather than subjective response quality or generic benchmark scores.

## 🎯 Project Overview

This project benchmarks two LLMs on their ability to convert complex natural-language questions into executable SQL queries.

The evaluation uses a **4-season Indian Premier League (IPL) SQLite database** containing real-world structured data and a curated dataset of 20 challenging SQL questions, ranging from Hard to Brutal difficulty.

Each generated SQL query is executed against the database, and its output is compared with a verified reference result to determine whether the model produces the correct answer.

## ⚙️ How It Works

The evaluation pipeline follows five steps:

1. **Curated Golden Dataset:** 20 complex natural-language questions paired with verified gold-standard SQL queries.
2. **LLM Query Generation:** Each model receives the database schema and question and generates an SQLite-compatible SQL query.
3. **Live Database Execution:** Generated SQL queries are executed directly against the IPL SQLite database.
4. **Automated Result Verification:** Actual query results are compared with expected gold-standard results to measure execution accuracy.
5. **Performance Tracking:** The pipeline records execution accuracy, average query latency, and total API cost.

## 📊 Benchmark Results

Both models were evaluated on the same 20 Hard-to-Brutal SQL questions.

| Evaluation Metric | Gemini Flash Lite | GPT-5.6 Terra |
|---|---:|---:|
| Execution Accuracy | **85.0% (17/20)** | 80.0% (16/20) |
| Average Latency | 6.70 seconds | **3.23 seconds** |
| Total API Cost | **$0.00662** | $0.06136 |

### 🏆 Key Findings

- **Accuracy:** Gemini Flash Lite achieved 85% execution accuracy, outperforming GPT-5.6 Terra by 5 percentage points.
- **Speed:** GPT-5.6 Terra was approximately 2.1x faster, with an average response time of 3.23 seconds.
- **Cost Efficiency:** Gemini Flash Lite was approximately 9.3x cheaper across all 20 queries.

### 📈 Performance Comparison

| Category | Winner |
|---|---|
| Execution Accuracy | Gemini Flash Lite |
| Query Generation Speed | GPT-5.6 Terra |
| Cost Efficiency | Gemini Flash Lite |

**Overall Observation:** Gemini Flash Lite delivered better accuracy at significantly lower cost, while GPT-5.6 Terra offered faster query generation. The results demonstrate that a more expensive model does not necessarily produce better SQL execution accuracy.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Database:** SQLite
- **AI Models:** GPT-5.6 Terra, Gemini Flash Lite
- **Core Concepts:** Text-to-SQL, LLM Evaluation, Automated Benchmarking
- **Evaluation Metrics:** Execution Accuracy, Query Latency, API Cost

## 📁 Project Structure

```text
LLM-Text-to-SQL-Benchmark/
│
├── dataset/
│   └── golden_dataset.json
│
├── database/
│   └── ipl_database.db
│
├── src/
│   ├── llm_runner.py
│   ├── sql_executor.py
│   └── evaluator.py
│
├── results/
│   └── benchmark_results.csv
│
├── .env.example
├── requirements.txt
└── README.md
```

*Note: Adjust the structure above to match your actual repository files.*

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/LLM-Text-to-SQL-Benchmark.git

cd LLM-Text-to-SQL-Benchmark
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root and add the required API credentials.

```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=
