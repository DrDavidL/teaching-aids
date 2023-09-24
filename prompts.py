chain_of_density_summary_template = """
Instructions:
- Context: Use only the {context} provided. Do not reference external sources.
- Task: Generate increasingly concise, entity-dense summaries for the context provided fitting the specified word count: {word_count}
- Repeat the following process 5 times:
  1. From the full context, identify 1-3 informative entities that are missing from the previously generated summary. These entities should be delimited by ';'.
  2. Write a denser summary of identical length that includes every detail from the previous summary and the newly identified missing entities.

Entity Definition:
- Relevant: Pertains to the main story.
- Specific: Descriptive yet concise (5 words or fewer).
- Novel: Not present in the previous summary.
- Faithful: Derived from the context.
- Location: Can be anywhere in the context.

Guidelines:
- The initial summary should be the specified words. It should be non-specific, with verbosity and fillers like 'this context discusses'.
- Every word in the summary should convey information. Enhance the previous summary for better flow and to accommodate additional entities.
- Optimize space by fusing information, compressing details, and eliminating uninformative phrases.
- Summaries should be dense, concise, and self-contained, ensuring they are comprehensible without referencing the context.
- Newly identified entities can be placed anywhere in the updated summary.
- Maintain all entities from the previous summary. If space constraints arise, incorporate fewer new entities.
- Ensure each summary has the same word count.

Output Format:
Your response should be in a structured format, comprising 2 lists; "Context-Specific Assertions" and "Assertions for General Use" These are followed by the final summary iteration,  
"Summary".
"""

ask_question_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

mcq_generation_template = """Generate {num_mcq} multiple choice questions for the context provided: {context} 
Include and explain the correct answer after the question. Apply best educational practices for MCQ design:
1. **Focus on a Single Learning Objective**: Each question should target a specific learning objective. Avoid "double-barreled" questions that assess multiple objectives at once.
2. **Ensure Clinical Relevance**: Questions should be grounded in clinical scenarios or real-world applications. 
3. **Avoid Ambiguity or Tricky Questions**: The wording should be clear and unambiguous. Avoid using negatives, especially double negatives. 
4. **Use Standardized Terminology**: Stick to universally accepted medical terminology. 
5. **Avoid "All of the Above" or "None of the Above"**
6. **Balance Between Recall and Application**: While some questions might test basic recall, strive to include questions that assess application, analysis, and synthesis of knowledge.
7. **Avoid Cultural or Gender Bias**: Ensure questions and scenarios are inclusive and don't inadvertently favor a particular group.
8. **Use Clear and Concise Language**: Avoid lengthy stems or vignettes unless necessary for the context. The complexity should come from the medical content, not the language.
9. **Make Plausible**: All options should be homogeneous and plausible to avoid cueing to the correct option. Distractors (incorrect options) are plausible but clearly incorrect upon careful reading.
10. **No Flaws**: Each item should be reviewed to identify and remove technical flaws that add irrelevant difficulty or benefit savvy test-takers.

Expert: Instructional Designer
Objective: To optimize the formatting of a multiple-choice question (MCQ) for clear display in a ChatGPT prompt.
Assumptions: You want the MCQ to be presented in a structured and readable manner for the ChatGPT model.

**Sample MCQ - Follow this format**:

**Question**:
What is the general structure of recommendations for treating Rheumatoid Arthritis according to the American College of Rheumatology (ACR)?

**Options**:
- **A.** Single algorithm with 3 treatment phases irrespective of disease duration
- **B.** Distinction between early (≤6 months) and established RA with separate algorithm for each
- **C.** Treat-to-target strategy with aim at reducing disease activity by ≥50%
- **D.** Initial therapy with Methotrexate monotherapy with or without addition of glucocorticoids


The correct answer is **B. Distinction between early (≤6 months) and established RA with separate algorithm for each**.

**Rationale**:

1. The ACR guidelines for RA treatment make a clear distinction between early RA (disease duration ≤6 months) and established RA (disease duration >6 months). The rationale behind this distinction is the recognition that early RA and established RA may have different prognostic implications and can respond differently to treatments. 
   
2. **A** is incorrect because while there are various treatment phases described by ACR, they don't universally follow a single algorithm irrespective of disease duration.

3. **C** may reflect an overarching goal in the management of many chronic diseases including RA, which is to reduce disease activity and improve the patient's quality of life. However, the specific quantification of "≥50%" isn't a standard adopted universally by the ACR for RA.

4. **D** does describe an initial approach for many RA patients. Methotrexate is often the first-line drug of choice, and glucocorticoids can be added for additional relief, especially in the early phase of the disease to reduce inflammation. But, this option does not capture the overall structure of ACR's recommendations for RA.

"""

clinical_trial_template = """Instructions:
- **Context**: Use only the {context} provided, which describes the clinical trial. Do not reference external sources.
- **Task**: Generate an increasingly detailed critical appraisal of the clinical trial provided, fitting the specified word count: {word_count}
- Repeat the following process 5 times:
  1. From the full context, identify 1-3 critical appraisal criteria or findings that are missing from the previously generated appraisal. These criteria or findings should be delimited by ';'.
  2. Write a more detailed appraisal of identical length that includes every detail from the previous appraisal and the newly identified missing criteria or findings.

Criteria Definition:
- **Relevant**: Pertains to the main objectives and methodology of the clinical trial.
- **Specific**: Descriptive yet concise (5 words or fewer).
- **Novel**: Not present in the previous appraisal.
- **Faithful**: Derived from the context.
- **Location**: Can be anywhere in the context.

Guidelines:
- The initial appraisal should be the specified words. It should be non-specific, with verbosity and fillers like 'this trial examines'.
- Every word in the appraisal should convey critical insight. Enhance the previous appraisal for better flow and to accommodate additional criteria or findings.
- Optimize space by fusing information, compressing details, and eliminating uninformative phrases.
- Appraisals should be dense, concise, and self-contained, ensuring they are comprehensible without referencing the context.
- Newly identified criteria or findings can be placed anywhere in the updated appraisal.
- Maintain all criteria or findings from the previous appraisal. If space constraints arise, incorporate fewer new criteria or findings.
- Ensure each appraisal has the same word count. Only output the final appraisal.

Output Format:
Your response should be in a structured format, comprising 2 lists; "Trial-Specific Critiques" and "General Clinical Trial Concerns". These are followed by the final appraisal output, "Critical Appraisal".
"""