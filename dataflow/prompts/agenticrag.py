class AutoPromptGeneratorPrompt:
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        pass

    def auto_prompt_generator_prompt(self, seed_data: str) -> str:
        system_prompt = f'''You will be given a piece of seed data, which may consist of a paragraph, dialogue, or any other form of text containing potential question-answer information.
Your task is to analyze this seed data carefully and generate a clear and effective prompt that can be used to instruct a language model to extract a single high-quality question-answer (QA) pair suitable for reinforcement learning (RL) training from this piece of data.

The generated prompt should:
Clearly describe the type and format of input the model will receive;
Explicitly ask for the extraction of a relevant QA pair;
Optionally include instructions about the desired style, level of detail, or coverage;
Be written in natural, precise English that could be directly used with another LLM;
Be strictly the prompt used to extract QA pairs, not the QA pairs themselves. 

Your prompts should contain the following instructions:
The question should be clear, focused, and unambiguous, such that it targets specific factual content from the input;
The answer should be a few words that are concise, factual and directly verifiable from the source rather than a whole sentence, enabling accurate reward computation in the RL pipeline;
Both the question and answer should be simple enough to facilitate evaluation and automatic feedback.

Don't include any additional explanations or comments in your output.
Don't repeat the seed data in your output.
Don't output the formatting instructions, just the prompt itself.
Here is the seed data you need to analyze and generate a prompt for:\n{seed_data}'''

        return system_prompt


class QAScorerPrompt:
    '''
    The prompt for the RAG scorer.
    '''
    def __init__(self):
        pass

    def question_quality_prompt(self) -> str:
        system_prompt = '''You are an expert question quality evaluator. Given a single question from a QA dataset, your job is to assess the **clarity and meaningfulness** of the question. Specifically, judge whether the question is clearly defined, unambiguous, and worth asking in a real-world or task-specific context.

Assign a score from 1 to 5 based on the following rubric:
5 = Very clear and meaningful question, well-posed  
4 = Clear but slightly underspecified or too general  
3 = Somewhat unclear or poorly scoped, but understandable  
2 = Ambiguous, vague, or unnatural  
1 = Nonsensical or meaningless

Output format:
**Grading**: [1-5]

**Feedback**: Explain your score. Mention if the question is ambiguous, overly broad, or lacks practical purpose. Suggest how to improve clarity or specificity if needed.

'''

        return system_prompt

    def answer_alignment_prompt(self) -> str:
        system_prompt = '''You are a response alignment evaluator. Your task is to assess whether a given answer **directly and clearly addresses the given question**.

Assign a score from 1 to 5 based on the following rubric:
5 = Fully and directly answers the question  
4 = Mostly addresses the question, with minor gaps or irrelevant additions  
3 = Partially answers the question but omits key aspects  
2 = Barely addresses the question or is off-topic  
1 = Completely unrelated to the question

Output format:
**Grading**: [1-5]

**Feedback**: Justify your score. Point out if the answer is evasive, incomplete, or misaligned. Suggest ways to better match the response to the question.

'''

        return system_prompt

    def answer_verifiability_prompt(self) -> str:
        system_prompt = '''You are an evaluator tasked with assessing how **easily verifiable** an answer is. You must determine whether the correctness of the answer can be **conveniently and unambiguously judged** — for example, whether it is fact-based, precise, and not subjective or vague.

Assign a score from 1 to 5 based on the following rubric:
5 = Very easy to verify; answer is objective, concrete, and unambiguous  
4 = Mostly verifiable, with minor ambiguities  
3 = Verifiable in parts, but some subjectivity or fuzziness  
2 = Hard to verify; answer is vague, speculative, or opinion-based  
1 = Unverifiable or meaningless

Output format:
**Grading**: [1-5]

**Feedback**: Explain your score. Identify elements that make verification easier or harder. Suggest rephrasing or grounding techniques to improve verifiability.

'''

        return system_prompt

    def downstream_value_prompt(self) -> str:
        system_prompt = '''You are a task relevance evaluator. Given a QA pair, assess how well this data point could **support a downstream task** such as classification, dialogue, retrieval, summarization, or knowledge grounding.

Assign a score from 1 to 5 based on the following rubric:
5 = Highly valuable for downstream tasks; question and answer are precise and informative  
4 = Useful with minor limitations  
3 = Moderately helpful; limited in informativeness or specificity  
2 = Of little value; vague or too generic to help the model learn  
1 = Useless or irrelevant for any downstream learning objective

Output format:
**Grading**: [1-5]

**Feedback**: Describe how the QA pair does or does not benefit potential downstream tasks. If relevant, suggest how to make it more useful for training.

'''

        return system_prompt

class AtomicTaskGeneratorPrompt:
    '''
    The prompt for the AtomicTaskGenerator.
    '''
    def __init__(self):
        pass
    
    def initial_conclusion_system_prompt(self) -> str:
        system_prompt = '''
        |
  # Conclusion Extraction and Relationship Generation Specifications

  ## I. Input/Output Requirements
  **Input**: Any document fragment  
  **Output**: JSON array where each element contains `conclusion` and `R` fields

  ## II. Conclusion Extraction Rules
  1. **Atomicity**  
      - Each conclusion must be an indivisible basic fact  
      - ✖ Prohibited combined conclusions: "A increased by 5% and B decreased by 2%" → Should be split into two conclusions

  2. **Verifiability**  
      - Must contain at least one definite identifier:  
        ✓ Numeric value (59.0%)  
        ✓ Time (2025/04/28)  
        ✓ Unique name (Humpback65B)  
      - ✖ Reject vague expressions: "Performance has improved"

  3. **Timeliness Handling**  
      - Explicitly mark time ranges when containing time-sensitive information  
      - Examples:  
        ✓ "Global GDP grew by 3.0% in 2023"  
        ✖ "Recent GDP growth of 3.0%"

  4. **Citation Integrity**  
      - If a conclusion cites other content (e.g., "as stated in (2)"), the complete content of (2) must be embedded in the conclusion

  ## III. Relationship (R) Generation Standards
  ### Attribute Requirements
  - **Structured**: Use semicolons to separate multi-metrics (Example 3)  
  - **Operational**: Directly usable for database queries or calculations  
    ✓ "City with the highest temperature"  
    ✖ "Conclusions about temperature"

  ### Generation Templates
  | Conclusion Type         | R Template                            | Example                         |
  |-------------------------|---------------------------------------|---------------------------------|
  | Single Numeric Result   | "[Indicator Name]"                    | A: "59.0%" → R: "Accuracy"      |
  | Comparative Conclusion  | "[Indicator] compared to [baseline] in [change dimension]" | A: "4.2% higher than baseline" → R: "Improvement in accuracy compared to baseline" |
  | Multi-dimensional Result| "[Primary Indicator] and its [sub-dimension] distribution" | A: "Average 59% (Humanities 65.6%)" → R: "Average accuracy and subject distribution" |

  ## IV. Output Specifications and Examples
  [
    {
      "conclusion": "Humpback65B achieved a zero-shot accuracy of 59.0% in the MMLU evaluation",
      "R": "Humpback65B's zero-shot accuracy"
    },
    {
      "conclusion": "On 2025/04/28, the closing price of XL Er Nantes-U was $11.34 (up 14.0%)",
      "R": "Closing price and percentage increase of XL Er Nantes-U on 2025/04/28"
    },
    {
        "conclusion": "90% of 27 million metric tons",
        "R": "Proportion of new global LNG supply from North America in 2025"
    },
    {
        "conclusion": "Abstract",
        "R": "Indexed part of Springer articles in databases"
    },
    {
        "conclusion": "2024-03-06",
        "R": "Publication date of Psychology Top 100 of 2023"
    },
    {
        "conclusion": "2018-01",
        "R": "Collection date of 'The Importance of Referencing - PMC'"
    },
    {
        "conclusion": "30-40%",
        "R": "Percentage of science report dedicated to results section"
    },
    {
        "conclusion": "$500 billion",
        "R": "Projected economic contribution of hybrid work models by 2030"
    },
    {
        "conclusion": "650,000+",
        "R": "Number of youth insights in India Skills Report 2025"
    },
    {
        "conclusion": "July 2024 issue",
        "R": "Consumer Reports publication in July 2024"
    },
    {
        "conclusion": "16th annual, 2024-12",
        "R": "Edition and publication date of Deloitte's Tech Trends 2025"
    },
    {
        "conclusion": "January 2024 issue",
        "R": "Consumer Reports publication in January 2024"
    },
    {
        "conclusion": "December 2021 issue",
        "R": "Consumer Reports publication in December 2021"
    },
    {
        "conclusion": "November 2021 issue",
        "R": "Consumer Reports publication in November 2021"
    },
    {
        "conclusion": "14",
        "R": "Death count in listeria outbreak linked to frozen shakes"
    },
    {
        "conclusion": "$122 million",
        "R": "Mega Millions jackpot amount for May 16"
    },
    {
        "conclusion": "32",
        "R": "Number of consecutive years United Way met its goal"
    },
    {
        "conclusion": "62%",
        "R": "Percentage increase in Chemical Sciences article submissions (2014-2021)"
    },
    {
        "conclusion": "11 pounds of fish",
        "R": "Fish trade for Europa League semifinal ticket"
    },
    {
        "conclusion": "2-1",
        "R": "PSG vs. Arsenal match result (Champions League)"
    }
  ]
        '''
        return system_prompt
    
    def initial_conclusion_prompt(self, input) -> str:
        prompt = f'''
    The document content to be processed is as follows: {input}
    '''
        return prompt
    
    def initial_question_system_prompt(self) -> str:
        system_prompt = '''Your task is to generate a corresponding question (Q) based on the given task identifier (ID), relationship (R), and answer (A).

  Input/Output Specifications:
  Input:
  - ID: Data source or query scope
  - R: Logical relationship for extracting the answer from the data
  - A: Known correct answer

  Output:
  - Must be in strict JSON format: {"Q": "generated question"}
  - No explanations or extra fields allowed

  Q must satisfy:
  1. Be a complete natural language question
  2. Allow deriving answer A by applying R after accessing context via ID

  Question Generation Principles:
  1. Exact correspondence - Each question must fully base on the original conclusion, with the answer being its core content.
  2. Derivability - The original conclusion must be directly derivable from the question and be the only correct answer.
  3. Self-containment - Questions must be complete and independent, not relying on external references or unspecified context.
  4. Information hiding - Do not reveal specific sources or data paths, but can include search hints.
  5. Specificity and clarity - Questions should include details like specific times to ensure unique answers.
  6. Single question - Generate only one question per conclusion.
  7. If the conclusion can only be obtained from input content, include hints via data source identifiers in the question.
  8. Language consistency - The language of each question must be the same as the conclusion's language.

  Examples:
  Input:
  ID: Global daily maximum temperatures
  R: City with the highest temperature
  A: xx City
  Output: {"Q": "What is the city with the highest temperature in global daily maximum temperatures?"}

  Input:
  ID: AASTOCKS Financial News - Key News
  R: Closing price and percentage increase of XL Er Nantes-U at 2025/04/28 04:30 GMT+010
  A: AASTOCKS Financial News - Key News reported that XL Er Nantes-U closed at $11.34 with a 14.0% increase. (Published at 2025/04/28 04:30 GMT+010)
  Output: {"Q": "What were the closing price (in USD) and percentage increase of XL Er Nantes-U at 2025/04/28 04:30 GMT+010?"}
  Example Explanation: Since this conclusion can be obtained from multiple financial websites, the question can omit the data source identifier.

  Input:
  ID: SELF-ALIGNMENT WITH INSTRUCTION BACKTRANSLATION
  R: Humpback65B's indicators in MMLU evaluation: 1) Zero-shot average accuracy; 2) Subfield scores (Humanities, STEM, Social Sciences, Others); 3) Comparison with LLaMA65B's zero-shot accuracy; 4) Few-shot (5-shot) score
  A: In the MMLU evaluation, the fine-tuned Humpback65B model achieved a zero-shot average accuracy of 59.0% (specific field scores: Humanities 65.6%, STEM 47.6%, Social Sciences 68.1%, Others 60.8%), an improvement over the base LLaMA65B model's zero-shot accuracy of 54.8%, although it performed worse in the few-shot (5-shot) setting (63.4%).
  Output: {"Q": "According to the paper 'SELF-ALIGNMENT WITH INSTRUCTION BACKTRANSLATION', what is the zero-shot average accuracy of the fine-tuned Humpback65B model in the MMLU evaluation, along with its specific field scores (Humanities, STEM, Social Sciences, and Others), how does this compare to the zero-shot accuracy of the base LLaMA65B model, and what score did Humpback65B achieve in the few-shot (5-shot) setting?"}
  Example Explanation: Since this conclusion is a specific result from a paper, the question includes the data source identifier as a hint.

  Only output JSON without additional content.
  '''
        return system_prompt
    
    def initial_question_prompt(self, conclusion, relation) -> str:
        prompt = f'''
            Data to be Processed:
        ID: text
        R: {relation}
        A: {conclusion}
        '''

        return prompt
    
    def clean_qa_system_prompt(self) -> str:
        system_prompt = '''Processing Rules:
  1. Extract ONLY the exact information requested in the question
  2. Preserve the original index numbering
  3. Never omit essential information
  4. Standardize all numerical formats:
      - Percentages: 8% (not "8percent" or "eight percent")
      - Numbers: Use commas for thousands (3,045)
      - Currency: $1,000 (not "1000 dollars")
      - Dates: YYYY-MM-DD format
      - Units: include (5kg, 10cm, etc.)

  Example:
  {
      "question": "How many travel trends for 2022 does '2025 Annual Travel Trends Report' present?",
      "original_answer": "The Neo4j graph database was used to organize 3,045 Raman spectra of exosomes.",
      "refined_answer": "3,045"
  }

  Required JSON format:
  {
      "question": str,
      "original_answer": str,
      "refined_answer": str
  }

  Key requirements:
  - Be extremely concise in refined_answer
  - Never add information not present in original_answer
  - Preserve all numerical values exactly
  - If question asks for specific data, extract only that data
  '''
        return system_prompt
    
    def clean_qa_prompt(self, input_qa) -> str:
        prompt = f'''
            The data need to be processed is as follows: {input_qa}
        '''

        return prompt