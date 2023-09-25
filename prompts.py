chain_of_density_summary_template = """**Instructions**:
- **Context**: Rely solely on the {context} given. Avoid referencing external sources.
- **Task**: Produce a series of summaries for the provided context, each fitting a word count of {word_count}. Each summary should be more entity-dense than the last.
- **Process** (Repeat 5 times):
  1. From the entire context, pinpoint 1-3 informative entities absent in the last summary. Separate these entities with ';'.
  2. Craft a summary of the same length that encompasses details from the prior summary and the newly identified entities.

**Entity Definition**:
- **Relevant**: Directly related to the main narrative.
- **Specific**: Descriptive but succinct (maximum of 5 words).
- **Novel**: Absent in the preceding summary.
- **Faithful**: Extracted from the context.
- **Location**: Can appear anywhere within the context.

**Guidelines**:
- Start with a general summary of the specified word count. It can be broad, using phrases like 'the context talks about'.
- Every word in the summary should impart meaningful information. Refine the prior summary for smoother flow and to integrate added entities.
- Maximize space by merging details, condensing information, and removing redundant phrases.
- Summaries should be compact, clear, and standalone, ensuring they can be understood without revisiting the context.
- You can position newly identified entities anywhere in the revised summary.
- Retain all entities from the prior summary. If space becomes an issue, introduce fewer new entities.
- Each summary iteration should maintain the designated word count.

**Output Format**:
Present your response in a structured manner, consisting of two sections: "Context-Specific Assertions" and "Assertions for General Use". Conclude with the final summary iteration under "Summary".
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