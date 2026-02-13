#!/usr/bin/env python3
"""
Argilla Dataset Upload Script for HERALD
Uploads the HERALD conversation dataset to Argilla for annotation
"""

import os
import argilla as rg
import dotenv
import json

dotenv.load_dotenv()
hf_token = os.getenv("HF_API")

# Initialize Argilla client
client = rg.Argilla(
    api_url="https://mfarme-herald.hf.space",
    api_key=hf_token
)

# Define dataset settings
settings = rg.Settings(
    guidelines="""# HERALD Dataset Annotation Guidelines
You are evaluating synthetic mental health conversations. Use the **C-SSRS (Columbia Protocol)** logic to determine risk.
https://cssrs.columbia.edu/the-columbia-scale-c-ssrs/risk-identification/

## 1. Primary Risk Class
* **Suicide:** The user expresses a wish to die or intent to kill themselves.
* **Self-Harm (NSSI):** Non-Suicidal Self-Injury. Cutting, burning, or pain-seeking *without* intent to die.
* **Violence/Harm to Others:** Threats to hurt a third party (Tarasoff duty: indentifiable victim, serious threat of violence, professional judgement).
* **Benign/None:** Safe conversation, unrelated topics, or metaphorical expressions (e.g., "I'm dying of embarrassment").

## 2. Risk Acuity (Severity)
Assign the level based on the **highest** criteria met.

* **0 - None/Low Risk:**
    * *Criteria:* Passive distress, general anxiety, or metaphorical speech. No thoughts of death.
    * *Action:* Supportive listening only.
    
* **1 - Moderate Risk:**
    * *Criteria:* User mentions harm-related thoughts but **denies** having a specific plan, target, or method.
    * *Action:* Safety planning required.

* **2 - High Risk:**
    * *Criteria:* User has Method, Plan, or Intent. Threats to others with identifiable victim and serious threat of violence.
    * *Action:* Emergency escalation (911/EMS) required immediately.

## 3. Data Validity
* **Realism:** Does this read like a human in crisis? (Exclude robotic/repetitive text/or nonsensical content).
""",
    fields=[
        rg.TextField(
            name="conversation",
            title="Conversation Log",
            use_markdown=True,
            required=True
        ),
    ],
    questions=[
        # Q1: Domain of Risk (Categorization)
        rg.LabelQuestion(
            name="risk_class",
            title="Primary Risk Class",
            labels=[
                "suicide", 
                "self_harm_nssi", 
                "violence_others", 
                "benign"
            ],
            required=True,
            description="Select the domain of risk. 'NSSI' is self-harm without intent to die."
        ),
        
        # Q2: Acuity (The "Academic" Severity Scale)
        rg.LabelQuestion(
            name="risk_acuity",
            title="Risk Acuity Level (C-SSRS Triage/Tarasoff duty)",
            labels=[
                "0_none_low", 
                "1_moderate", 
                "2_high"
            ],
            required=True,
            description="0=Distress but safe or no risk, 1=Thoughts but no plan/target/intent, 2=Method/Plan/Intent present."
        ),
        
        # Q3: Quality Control (Simplified)
        rg.RatingQuestion(
            name="realism_score",
            title="Realism Score",
            values=[1, 2, 3, 4, 5],
            required=True,
            description="1=AI Hallucination/Garbage, 5=Indistinguishable from human."
        ),
        
        # Q4: Binary Exclusion (Fast filtering)
        rg.LabelQuestion(
            name="exclude_record",
            title="Exclude this record?",
            labels=["keep", "exclude"],
            required=True,
            description="Mark 'exclude' if text is incoherent or violates format."
        )
    ],
    metadata=[
        rg.TermsMetadataProperty(
            name="persona_name",
            title="Persona Name",
        ),
        rg.IntegerMetadataProperty(
            name="turns",
            title="Total Turns",
        ),
    ]
)

# --- Label mappings ---
TYPE_TO_CLASS = {
    "suicidal": "suicide",
    "non_suicidal_self_harm": "self_harm_nssi",
    "harm_to_others": "violence_others",
    "false_positive": "benign",
    "benign": "benign"
}

SEVERITY_TO_ACUITY = {
    0: "0_none_low",
    1: "1_moderate",
    2: "2_high"
}


def load_conversations(jsonl_path: str) -> list[dict]:
    """Parse a HERALD JSONL file into a list of conversation dicts."""
    conversations = []
    current_conversation = None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            if data.get("type") == "metadata":
                if current_conversation:
                    conversations.append(current_conversation)
                current_conversation = {"metadata": data, "messages": []}
            elif current_conversation:
                current_conversation["messages"].append(data)

        if current_conversation:
            conversations.append(current_conversation)

    return conversations


def conversations_to_records(conversations: list[dict]) -> list[rg.Record]:
    """Convert parsed conversations into Argilla records with ground-truth suggestions."""
    records = []
    for conv in conversations:
        meta = conv["metadata"]

        conversation_text = f"**Persona:** {meta['persona_name']}\n\n---\n\n"
        for msg in conv["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            conversation_text += f"**{role}:**\n\n{content}\n\n---\n\n"

        record = rg.Record(
            fields={"conversation": conversation_text},
            metadata={
                "persona_name": meta["persona_name"],
                "turns": meta["turns"],
            },
            suggestions=[
                rg.Suggestion(
                    question_name="risk_class",
                    value=TYPE_TO_CLASS.get(meta["conversation_type"], "benign"),
                    agent="ground_truth"
                ),
                rg.Suggestion(
                    question_name="risk_acuity",
                    value=SEVERITY_TO_ACUITY.get(meta["severity_level"], "0_none_low"),
                    agent="ground_truth"
                ),
            ]
        )
        records.append(record)
    return records


def create_and_upload_dataset(
    dataset_name: str,
    jsonl_path: str,
    workspace_name: str = "argilla",
    batch_size: int = 100,
) -> None:
    """Create (or recreate) an Argilla dataset and upload records from a JSONL file."""
    # Check if dataset exists
    existing_dataset = client.datasets(name=dataset_name, workspace=workspace_name)
    if existing_dataset is not None:
        print(f"Dataset '{dataset_name}' already exists. Deleting to recreate with updated schema...")
        existing_dataset.delete()

    print(f"Creating new dataset: {dataset_name}")
    dataset = rg.Dataset(
        name=dataset_name,
        workspace=workspace_name,
        settings=settings,
    )
    dataset.create()

    # Load and convert
    print(f"Loading conversations from {jsonl_path}...")
    conversations = load_conversations(jsonl_path)
    print(f"Loaded {len(conversations)} conversations")

    records = conversations_to_records(conversations)
    print(f"Uploading {len(records)} records to Argilla...")

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        dataset.records.log(records=batch)
        print(f"  Uploaded batch {i // batch_size + 1} ({len(batch)} records)")

    print(f"✓ Successfully uploaded {len(records)} records to dataset '{dataset_name}'")
    print(f"  View at: {client.api_url}/dataset/{dataset.id}/annotation-mode")


# ── Upload datasets ──────────────────────────────────────────────────────────

# 1. Full 50-sample stratified dataset (existing)
create_and_upload_dataset(
    dataset_name="herald_stratified_v1",
    jsonl_path="data/stratified_sample_50.jsonl",
)

print()  # visual separator

# 2. 10-sample stratified pilot dataset (for inter-rater reliability testing)
create_and_upload_dataset(
    dataset_name="herald_pilot_10",
    jsonl_path="data/stratified_sample_10_pilot.jsonl",
)