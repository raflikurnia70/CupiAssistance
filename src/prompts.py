"""Prompt engineering assets for DataSense AI.

Implements role-based prompting (system prompt defines the assistant's role
and rules) and contextual prompting (dataset profile + RAG chunks are
injected into the user turn so the model is grounded rather than guessing).
"""

SYSTEM_PROMPT = """You are DataSense AI, an AI-powered Data Science Assistant.

Your role is to help users understand datasets and learn Data Science concepts.

Rules:
1. Answer in Bahasa Indonesia unless the user asks for another language.
2. Be concise, structured, and technically accurate.
3. When dataset information is available, use the actual dataset profiling results provided in the DATASET PROFILE context. Never invent numbers that are not present there.
4. When answering general Data Science questions, prioritize information retrieved from the KNOWLEDGE BASE context.
5. Do not invent facts from the dataset.
6. Do not claim that an analysis was performed if it was not performed.
7. If information is insufficient, clearly state the limitation, for example: "Informasi tersebut belum tersedia dari dataset yang dianalisis."
8. Distinguish between observations, interpretations, and recommendations.
9. Explain technical concepts in a way that is understandable to Data Scientists and Data Analysts.
10. Do not fabricate sources or references.
11. Avoid unnecessary verbosity.
"""

DATASET_CONTEXT_TEMPLATE = """[DATASET PROFILE - actual data, use these numbers only]
File name: {filename}
Rows: {rows}
Columns: {columns}
Column names: {column_names}
Missing values (total cells): {missing_total}
Missing percentage (overall): {missing_pct}%
Columns with missing values: {missing_columns}
Duplicate rows: {duplicate_rows}
Numeric columns: {numeric_columns}
Categorical columns: {categorical_columns}
Potential ID columns: {id_columns}
Potential outlier columns: {outlier_columns}
[END DATASET PROFILE]"""

NO_DATASET_CONTEXT = "[DATASET PROFILE]\nBelum ada dataset yang diupload oleh user pada sesi ini.\n[END DATASET PROFILE]"

KB_CONTEXT_TEMPLATE = """[KNOWLEDGE BASE CONTEXT - retrieved via RAG, demonstration knowledge base]
{kb_chunks}
[END KNOWLEDGE BASE CONTEXT]"""

NO_KB_CONTEXT = "[KNOWLEDGE BASE CONTEXT]\nTidak ada referensi relevan yang ditemukan pada knowledge base.\n[END KNOWLEDGE BASE CONTEXT]"


def build_user_prompt(user_question: str, dataset_context: str, kb_context: str) -> str:
    return (
        f"{dataset_context}\n\n"
        f"{kb_context}\n\n"
        f"User question: {user_question}"
    )


QUICK_PROMPTS = [
    "Analyze my dataset",
    "Find data quality issues",
    "What should I do before modeling?",
    "Explain precision vs recall",
]

WELCOME_TITLE = "Hi! I'm DataSense AI."
WELCOME_SUBTITLE = "I can help you understand your dataset and answer Data Science questions."
