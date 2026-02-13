---
license: mit
task_categories:
- text-classification
- text-generation
tags:
- medical
- mental-health
- suicide-prevention
- harm-detection
- safety
- crisis-intervention
size_categories:
- 1K<n<10K
---

# Dataset Card for HERALD: Harm Evaluation and Risk Assessment for Language Detection

The conversation dataset was generated using the OpenAI SDK and OpenRouter API. Pydantic data models were created to define personas, conversation types, and severity level. The model utilized to generate the synthetic conversation was Qwen 3 235b (a22b-2507) with a temperature of 1.0. Seven individual personas were created including name, race, ethnicity, age, language, disability, religion, sex, occupation, interests, background, reaction to support, language style, and user experience (with LLMs).

## Dataset Details

### Dataset Description

This dataset contains synthetic mental health crisis conversations designed to evaluate large language models' ability to detect harm-related situations and make appropriate escalation decisions. 

Five conversation types were created including: 
1. **Suicidal ideation** (30%) - Explicit suicidal thoughts, plans, or expressions of wanting to die
2. **Non-suicidal self-harm** (25%) - Self-injury behaviors without suicidal intent (cutting, burning, etc.)
3. **Harm to others** (25%) - Threats, plans, or expressions of wanting to harm other people
4. **False positives** (15%) - Conversations that appear concerning but are actually benign (gaming references, metaphors, etc.)
5. **Benign** (5%) - Normal, safe conversations with no safety concerns

A system prompt was created outlining the generation model's instructions to role-play as the persona and generate a conversation consistent with their persona and defined conversation type.

To ensure the conversation between the persona and the AI assistant represented a realistic scenario, the AI assistant's turn did not include the system prompt or reveal any details about the persona, conversation type, or severity of the conversation. Each conversation was permitted 3 turns: user, AI assistant, user. Each conversation was stored in JSONL format with the associated labels for model evaluation.

- **Curated by:** Matthew S. Farmer, PhD RN
- **Language(s) (NLP):** English
- **License:** MIT

## Uses

### Direct Use

This dataset is intended for:
- Benchmarking and evaluating LLMs on harm detection capabilities
- Training and fine-tuning models for crisis intervention scenarios
- Research on AI safety in mental health applications
- Developing escalation protocols for LLM-based support systems
- Assessing model performance across different severity levels and risk categories

### Out-of-Scope Use

This dataset should NOT be used for:
- Direct deployment in mental health support systems without human oversight
- Training models without additional validation on real-world crisis data
- Making actual clinical decisions or replacing professional mental health assessment
- Evaluating models on non-English conversations
- Applications without appropriate safety protocols and emergency response capabilities

## Dataset Structure

The dataset is stored in JSONL format where each line contains either:

1. **Conversation metadata**:
```json
{
  "type": "metadata",
  "persona_name": "Persona Name",
  "conversation_type": "suicidal",
  "severity_level": 2,
  "turns": 3
}
```

2. **Message data**:
```json
{
  "role": "user",
  "content": "..."
}
```

### Data Fields

**Conversation Metadata:**
- `persona_name`: Name of the persona in the conversation
- `conversation_type`: One of five categories (suicidal, non_suicidal_self_harm, harm_to_others, false_positive, benign)
- `severity_level`: Risk level (0=no risk, 1=follow-up required, 2=immediate risk)
- `turns`: Number of conversation turns (always 3)

**Message Data:**
- `role`: "user" or "assistant"
- `content`: The message text

**Persona Attributes** (used during generation):
- Demographics: name, race, ethnicity, age, sex
- Background: occupation, interests, background story
- Characteristics: religion, disability status, language style
- Behavioral: reaction_to_support, user_experience with LLMs

### Data Splits

The dataset contains approximately 1,000+ conversations with the following target distribution:
- Suicidal ideation: 30%
- Non-suicidal self-harm: 25%
- Harm to others: 25%
- False positives: 15%
- Benign: 5%

## Dataset Creation

### Curation Rationale

This dataset was created to address a critical gap in LLM safety evaluation. With the increasing use of LLMs in mental health support applications, there is an urgent need to benchmark models on their ability to:
- Detect harm-related crisis situations
- Make appropriate escalation decisions
- Minimize false negatives in life-threatening scenarios
- Handle diverse demographic and linguistic expressions of crisis

The synthetic generation approach was chosen to enable large-scale evaluation while maintaining ethical standards around the use of real crisis communications.

### Source Data

#### Data Collection and Processing

The dataset was generated entirely synthetically using:
- **Generation Model:** Qwen 3 235b (a22b-2507)
- **Temperature:** 1.0 for diverse, naturalistic responses
- **Max Tokens:** 8000
- **Turn Structure:** Fixed 3-turn format (user → assistant → user)
- **API:** OpenRouter via OpenAI SDK
- **Data Models:** Pydantic for type safety and validation

The generation process involved:
1. Creating detailed persona profiles with diverse demographic characteristics
2. Defining conversation type prompts with specific guidelines for each risk category
3. Generating initial user messages based on persona and conversation type
4. Generating assistant responses without knowledge of the underlying labels
5. Generating follow-up user messages to create realistic multi-turn interactions
6. Storing conversations with complete metadata for evaluation

#### Who are the source data producers?

The source data is entirely synthetic, generated by Qwen 3 235b based on:
- 7 diverse personas created with varying demographics, backgrounds, and characteristics
- Structured prompts for 5 conversation categories
- System instructions for realistic role-playing
- No human-written crisis conversations were used in generation

### Annotations

#### Annotation process

All annotations are generated automatically during the conversation creation process:
- **Conversation Type:** Assigned prior to generation based on target distribution
- **Severity Level:** Determined by conversation type (0=benign/false positive, 1=low risk, 2=high risk)
- **Metadata:** Captured during generation (timestamps, turn counts, persona details)

No human annotation was performed. The labels represent the intended conversation characteristics programmed into the generation process.

#### Who are the annotators?

Not applicable - all labels are programmatically assigned during synthetic generation.

#### Personal and Sensitive Information

This dataset contains **no real personal or sensitive information**. All conversations are synthetic and generated by an AI model. The personas are fictional characters with fabricated backgrounds, demographics, and characteristics.

However, the dataset does contain:
- Simulated crisis language (suicide, self-harm, violence)
- Demographic attributes of fictional personas
- Scenarios depicting mental health crises

Users should be aware of the sensitive nature of the content and handle it appropriately in research and development contexts.

## Bias, Risks, and Limitations

**Limitations:**
- **Synthetic Data Only:** May not fully capture the complexity and unpredictability of genuine crisis communications
- **English Language Only:** Limited to English conversations, excluding multilingual and cross-cultural expressions of crisis
- **Three-Turn Limit:** May not reflect extended interactions typical in real mental health support
- **Generation Bias:** Potential biases from the Qwen 3 235b model may be present in conversations
- **Limited Personas:** Seven personas may not adequately represent the full spectrum of cultural, linguistic, and socioeconomic diversity
- **Simplified Severity Mapping:** Three-level severity classification may not capture nuanced risk levels

**Risks:**
- Over-reliance on this synthetic dataset without real-world validation could lead to unsafe deployments
- Models trained solely on this data may not generalize to actual crisis situations
- The false positive category may inadvertently teach models to dismiss legitimate concerns
- Performance on synthetic data may not predict performance on real crisis communications

**Biases:**
- Generation process may perpetuate biases present in the Qwen 3 235b training data
- Cultural and linguistic expressions of crisis may be underrepresented
- Demographic diversity limited by the seven predefined personas
- Assistant responses generated without specialized mental health training

### Recommendations

- **Do not deploy** models evaluated only on this dataset in real mental health support systems without extensive validation
- **Supplement with real-world data** when ethically feasible and properly de-identified
- **Implement human oversight** for any system using these models in crisis scenarios
- **Conduct fairness audits** across demographics to identify and address performance disparities
- **Validate findings** with mental health professionals and crisis intervention experts
- **Establish safety protocols** including emergency escalation pathways before any deployment
- **Monitor performance continuously** in real-world applications with rapid response to failures

## Citation

**Paper:**
Farmer, M. S. (2025). HERALD: Harm Evaluation and Risk Assessment for Language Detection. *Unpublished manuscript*, University of Missouri, Sinclair School of Nursing.

**BibTeX:**
```bibtex
@unpublished{farmer2025herald,
  author = {Farmer, Matthew S.},
  title = {HERALD: Harm Evaluation and Risk Assessment for Language Detection},
  year = {2025},
  note = {Unpublished manuscript},
  institution = {University of Missouri, Sinclair School of Nursing}
}
```

**APA:**
Farmer, M. S. (2025). *HERALD: Harm Evaluation and Risk Assessment for Language Detection* [Unpublished manuscript]. Sinclair School of Nursing, University of Missouri.

## Glossary

- **Escalation:** The process of referring a high-risk conversation to human professionals for immediate intervention
- **Severity Level:** Three-tier risk classification (0=no risk, 1=follow-up required, 2=immediate risk)
- **Conversation Type:** Category of harm or safety concern expressed in the conversation
- **Persona:** Fictional character profile with demographic and behavioral attributes used for generation
- **Turn:** A single message exchange in the conversation (user or assistant message)
- **False Positive:** Conversation that may appear concerning superficially but contains no actual risk
- **Synthetic Data:** Artificially generated conversations created by AI rather than real human interactions

## More Information

**Repository:** https://github.com/mfarme/herald

**Contact:** Matthew S. Farmer, PhD RN - University of Missouri, Sinclair School of Nursing

**Related Work:**
- Full evaluation results and analysis available in accompanying research paper
- Dataset generation code available in repository
- Evaluation framework and benchmarking tools included

**Content Warning:** This dataset contains discussions and simulations of suicide, self-harm, and violent language. Users should be prepared for sensitive mental health content.

## Dataset Card Authors

Matthew S. Farmer, PhD RN

## Dataset Card Contact

Matthew S. Farmer, PhD RN  
University of Missouri, Sinclair School of Nursing  
GitHub: [@mfarme](https://github.com/mfarme)
