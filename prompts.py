chain_of_density_summary = """
Instructions:
- Context: You will generate increasingly concise, entity-dense summaries of the {context} provided.
- Repeat the following process 5 times:
  1. From the context, identify 1-3 informative entities that are missing from the previously generated summary. These entities should be delimited by ';'.
  2. Write a denser summary of identical length that includes every detail from the previous summary and the newly identified missing entities.

Entity Definition:
- Relevant: Pertains to the main story.
- Specific: Descriptive yet concise (5 words or fewer).
- Novel: Not present in the previous summary.
- Faithful: Derived from the context.
- Location: Can be anywhere in the context.

Guidelines:
- The initial summary should be approximately {word_count} words. It should be non-specific, with verbosity and fillers like 'this context discusses'.
- Every word in the summary should convey information. Enhance the previous summary for better flow and to accommodate additional entities.
- Optimize space by fusing information, compressing details, and eliminating uninformative phrases.
- Summaries should be dense, concise, and self-contained, ensuring they are comprehensible without referencing the context.
- Newly identified entities can be placed anywhere in the updated summary.
- Maintain all entities from the previous summary. If space constraints arise, incorporate fewer new entities.
- Ensure each summary has the same word count.

Output Format:
Your response should be in a structured format, comprising a list of "Concepts Addressed" followed by the final summary iteration,  "Summary".
"""

mcq_template = """Answer the question based only on the following context:
{context}

Question: {faculty_question}
"""

mcq_generation = """Generate 3 multiple choice questions for the context provided. Include the correct answer after the question. Use practices for optimal MCQ design:
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

**Here is a sample MCQ. Follow this format**:
1. What is the general structure of recommendations for treating Rheumatoid Arthritis according to the American College of Rheumatology (ACR)?

Options:
A. Single algorithm with 3 treatment phases irrespective of disease duration
B. Distinction between early (≤6 months) and established RA with separate algorithm for each
C. Treat-to-target strategy with aim at reducing disease activity by ≥50%
D. Initial therapy with Methotrexate monotherapy with or without addition of glucocorticoids
Answer: B. Distinction between early (≤6 months) and established RA with separate algorithm for each
"""