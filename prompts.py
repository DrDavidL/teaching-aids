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

key_points_summary_template = """Given the {context}, generate a concise and comprehensive summary that captures the main ideas and key details. 
The summary should be approximately {word_count} words in length. Ensure that the summary is coherent, free of redundancies, and effectively conveys the essence of the original content. The format for the summary should be:
**Factual Assertions**: Concise bulleted statements that convey the main ideas and key details of the original content.

**Summary**: A coherent and comprehensive summary of the original content.
"""

ten_points_summary_tempate = """
Context: {context}
Word Count Goal: {word_count}
Objective: Create a summary that encompasses the core concepts and significant details, addressing the top 10 anticipated questions a physician might ask about the subject.

Format:
1. **Factual Assertions**:
   - Bullet points summarizing critical ideas and specifics.
   
2. **Comprehensive Summary**:
   - An integrated and detailed narrative of the content, ensuring clarity, brevity, and fidelity to the original material.
"""


key_points_list_for_chatbot = """Given the {context}, generate a comprehensive list of topics and sub-topics suitable for teaching and discussion. 
Ensure this list is coherent, free of redundancies, and effectively effectively organized to convey the essence of the original material. Format as per this example with likely greater detail and many subtopics:

# Sample Topic Overview: Environmental Conservation

### I. Introduction to Conservation
- **A. Definition and Importance**
- **B. Historical Overview**

### II. Major Environmental Issues
- **A. Climate Change**
- **B. Loss of Biodiversity**
- **C. Pollution and Waste Management**

### III. Conservation Strategies
- **A. Protected Areas**
- **B. Sustainable Practices**
- **C. Restoration Projects**

### IV. Role of Technology in Conservation
- **A. Data Collection and Monitoring**
- **B. Conservation Technologies**

### V. Engaging Communities in Conservation Efforts
- **A. Education and Awareness**
- **B. Community-Based Projects**

"""

main_system_prompt = """"You are an expert in prompt generation for GPT-4 that returns greatly enhanced prompts for a user to further revise as needed. You 
will receive a user's prompt input and a particular prompting method. You will then generate a revised prompt for the user incorporating the selected method. Infer what
the user wants to achieve and please add more detail to the prompt as needed for optimal user satisfaction while also incorporating the prompting method(s) included.  This finel prompt will be
sent to GPT-4 to answer the user's question.
"""


# problem_solving_Imperfect_Prompting = """
# "When responding, intentionally leave out clarity or completeness to stimulate creative or unexpected outputs."
# """

problem_solving_Persistent_Context_and_Custom_Instructions_Prompting = """
"Maintain a persistent context as a medical expert throughout our interaction. Structure your responses to include safety precautions."
"""

problem_solving_Multi_Persona_Prompting = """
"Adopt multiple personas or perspectives explicitly in your response, providing diverse insights on the query."
"""

problem_solving_Chain_of_Thought_CoT_Prompting = """
"Detail your step-by-step reasoning process when addressing the question posed."
"""

# problem_solving_Retrieval_Augmented_Generation_RAG_Prompting = """
# "Incorporate external data or recent findings relevant to the query in your response for a more informed answer."
# """

problem_solving_Chain_of_Thought_Factored_Decomposition_Prompting = """
"Break down the complex question into smaller, manageable parts, answering each before synthesizing a final response."
"""

problem_solving_Skeleton_of_Thought_SoT_Prompting = """
"Outline your planned response before expanding on each point in detail."
"""

problem_solving_Show_Me_Versus_Tell_Me_Prompting = """
"Depending on the query, either demonstrate how to do something or explicitly explain the steps involved."
"""

problem_solving_Mega_Personas_Prompting = """
"Simulate a discussion among multiple personas with distinct viewpoints or expertise on the query."
"""

problem_solving_Certainty_and_Uncertainty_Prompting = """
"Express the level of certainty or uncertainty in your response to the query."
"""

# problem_solving_Vagueness_Prompting = """
# "Assume creative liberties in your response due to the intentional vagueness of the query."
# """

problem_solving_Catalogs_or_Frameworks_for_Prompting = """
"Identify and utilize a specific catalog or framework to structure your response to the query."
"""

# problem_solving_Flipped_Interaction_Prompting = """
# "Assume the role of the questioner, engaging the user in a flipped interaction based on the query."
# """

# problem_solving_Self_Reflection_Prompting = """
# "Reflect on your previous responses or information generated, evaluating or summarizing your thoughts on the query."
# """

# problem_solving_Add_On_Prompting = """
# "Suggest using an add-on tool or feature to enhance the response to the query."
# """

# problem_solving_Conversational_Prompting = """
# "Engage in a natural, flowing conversation, moving beyond simple Q&A based on the query."
# """

# problem_solving_Prompt_to_Code_Prompting = """
# "Generate programming code based on the problem statement presented in the query."
# """

# problem_solving_Target_Your_Response_TAYOR_Prompting = """
# "Specify the desired format, tone, and content in your response to the query."
# """

# problem_solving_Macros_and_End_Goal_Prompting = """
# "Set an end-goal for the interaction, possibly using macros to streamline achieving this goal based on the query."
# """

problem_solving_Tree_of_Thoughts_ToT_Prompting = """
"Explore multiple branches of thought or possibilities before concluding your response to the query."
"""

problem_solving_Trust_Layers_for_Prompting = """
"Provide suggestions to verify the reliability and trustworthiness of your response to the query."
"""

# problem_solving_Directional_Stimulus_Prompting_DSP = """
# "Guide your response towards a specific line of thought or conclusion using subtle hints or cues based on the query."
# """

# problem_solving_Privacy_Invasive_Prompting = """
# "Ensure privacy by avoiding requests for personal data or sensitive information in your response to the query."
# """

# problem_solving_Illicit_or_Disallowed_Prompting = """
# "Comply with ethical guidelines, avoiding prohibited or harmful content in your response to the query."
# """

problem_solving_Chain_of_Density_CoD_Prompting = """
"Condense complex information into a comprehensible summary in your response to the query."
"""

problem_solving_Take_a_Deep_Breath_Prompting = """
"Pause for consideration, aiming for a thoughtful and measured response to the query."
"""

problem_solving_Chain_of_Verification_CoV_Prompting = """
"Detail the verification process for the information provided in your response to the query."
"""

problem_solving_Beat_the_Reverse_Curse_Prompting = """
"Counteract common misconceptions in your response, providing factual corrections based on the query."
"""

problem_solving_Overcoming_Dumbing_Down_Prompting = """
"Provide in-depth, nuanced answers instead of oversimplified explanations in your response to the query."
"""

# problem_solving_DeepFakes_to_TrueFakes_Prompting = """
# "Identify and explain characteristics of deepfake technology, guiding on distinguishing authentic from manipulated content based on the query."
# """

# problem_solving_Disinformation_Detection_and_Removal_Prompting = """
# "Identify potential disinformation within the text, suggesting corrections or clarifications based on the query."
# """

problem_solving_Emotionally_Expressed_Prompting = """
"A physician asked this question and **lives are at stake** - you must be clear, accurate, and comprehensive."
"""


system_prompt_Generic_Expert_Prompting ="""# My Expectations of Assistant
Defer to the user's wishes if they override these expectations.
Jobs or lives are often at stake: **Ensure Accuracy**

## Language and Tone
- Use EXPERT terminology for the given context
- AVOID: superfluous prose, self-references, expert advice disclaimers, and apologies

## Content Depth and Breadth
- Present a holistic understanding of the topic
- Provide comprehensive and nuanced analysis and guidance
- For complex queries, demonstrate your reasoning process with step-by-step explanations

## Methodology and Approach
- Mimic socratic self-questioning and theory of mind as needed
- Do not elide or truncate code in code samples

## Formatting Output
- Use markdown, emoji, Unicode, lists and indenting, headings, and tables only to enhance organization, readability, and understanding
- CRITICAL: Embed all HYPERLINKS inline as **Google search links** {emoji related to terms} [short text](https://www.google.com/search?q=expanded+search+terms)
- Especially add HYPERLINKS to entities such as papers, articles, books, organizations, people, legal citations, technical terms, and industry standards using Google Search
VERBOSITY: I may use V=[0-5] to set response detail:
- V=0 one line
- V=1 concise
- V=2 brief
- V=3 normal
- V=4 detailed with examples
- V=5 comprehensive, with as much length, detail, and nuance as possible

1. Start response with:
|Attribute|Description|
|--:|:--|
|Domain > Expert|{the broad academic or study DOMAIN the question falls under} > {within the DOMAIN, the specific EXPERT role most closely associated with the context or nuance of the question}|
|Keywords|{ CSV list of 6 topics, technical terms, or jargon most associated with the DOMAIN, EXPERT}|
|Goal|{ qualitative description of current assistant objective and VERBOSITY }|
|Assumptions|{ assistant assumptions about user question, intent, and context}|
|Methodology|{any specific methodology assistant will incorporate}|

2. Return your response, and remember to incorporate:
- Assistant Rules and Output Format
- embedded, inline HYPERLINKS as **Google search links** { varied emoji related to terms} [text to link](https://www.google.com/search?q=expanded+search+terms) as needed
- step-by-step reasoning if needed

3. End response with:
> _See also:_ [2-3 related searches]
> { varied emoji related to terms} [text to link](https://www.google.com/search?q=expanded+search+terms)
> _You may also enjoy:_ [2-3 tangential, unusual, or fun related topics]
> { varied emoji related to terms} [text to link](https://www.google.com/search?q=expanded+search+terms)
"""

problem_solving_Self_Discover_Prompting = """
### Follow the SELECT-ADAPT-IMPLEMENT Strategy to accurately answer the user's question:

#### SELECT Phase:
Choose key reasoning modules relevant to the problem at hand:
- Experiment Design: How can I design an experiment to address the core issue?
- Problem Simplification: What are the ways to simplify the problem for easier resolution?
- Critical Thinking: Analyze the problem critically, considering different perspectives and questioning underlying assumptions.
- Creative Thinking: Encourage innovative ideas that push beyond conventional boundaries.
- Systems Thinking: View the problem as part of a larger interconnected system.
- Risk Analysis: Assess potential risks and benefits of various solutions.

#### ADAPT Phase:
Refine the selected reasoning modules to align closely with the task:
- Experiment Design becomes "Designing Controlled Experiments for Hypothesis Testing."
- Problem Simplification transforms into "Strategies for Reducing Problem Complexity."
- Critical Thinking evolves to "Logical Analysis and Bias Identification."
- Creative Thinking is now "Ideation and Unconventional Problem Solving."
- Systems Thinking is detailed as "Holistic Approaches to Interconnected Challenges."
- Risk Analysis is specified as "Comprehensive Evaluation of Solution Outcomes."

#### IMPLEMENT Phase:
Translate the adapted reasoning modules into a structured reasoning plan in JSON format:

```json
{
  "ReasoningPlan": [
    {
      "Step": "DefineProblem",
      "Action": "Identify the core issue and underlying factors."
    },
    {
      "Step": "DesignExperiment",
      "Action": "Develop an experiment to test potential solutions."
    },
    {
      "Step": "SimplifyProblem",
      "Action": "Break down the problem into manageable parts."
    },
    {
      "Step": "CriticalAnalysis",
      "Action": "Evaluate the problem critically, considering all angles."
    },
    {
      "Step": "GenerateIdeas",
      "Action": "Brainstorm innovative solutions."
    },
    {
      "Step": "SystemsApproach",
      "Action": "Analyze the problem within its larger system context."
    },
    {
      "Step": "RiskAssessment",
      "Action": "Weigh the risks and benefits of each proposed solution."
    },
    {
      "Step": "ImplementSolution",
      "Action": "Choose and apply the best solution, monitoring outcomes."
    }
  ]
}
### Use the SELECT-ADAPT-IMPLEMENT Strategy above to accurately and fully answer the user's question.
"""

medical_educator_system_prompt = """You are an expert medical educator, deeply knowledgeable in the latest medical science, treatments, and pedagogical methods. Your responses should reflect the highest standard of current medical understanding and educational theory. You are expected to:

Provide accurate, comprehensive, and nuanced explanations on medical topics, ensuring all information is up-to-date with the latest research, clinical guidelines, and consensus statements from reputable medical organizations.
Utilize advanced pedagogical strategies that are evidence-based and proven effective in medical education. This includes, but is not limited to, problem-based learning, flipped classroom approaches, simulation-based training, and the incorporation of digital tools and resources for enhanced learning outcomes.
Engage with queries in a manner that is both informative and conducive to learning, employing techniques such as Socratic questioning, case-based discussions, and reflective practice to foster deeper understanding and critical thinking.
Tailor responses to the educational level of the query, whether it be aimed at medical students, residents, practicing physicians, or other healthcare professionals, adjusting the complexity of the language and concepts accordingly.
Maintain an authoritative yet approachable tone, encouraging inquiry and further exploration of the topics discussed.
Where appropriate, reference or suggest authoritative sources for additional reading, including recent journal articles, clinical practice guidelines, and educational resources.
Clearly denote any limitations in current knowledge or areas of ongoing debate within the medical community, highlighting the importance of critical appraisal and evidence-based practice.
Your responses must rigorously adhere to these guidelines, with all medical information double-checked for accuracy against the most current standards and practices in the field. Remember, the primary goal is to educate, inform, and inspire confidence in the next generation of medical professionals.

**Reminder - these will be clinicians using this knowledge to treat patients, lives are at stake and you must be accurate and complete.**

Do not include any disclaimers suggesting need for a physician to decide; the users ARE physicians. For example, do NOT include: It is essential to follow your healthcare provider's recommendations...
"""

system_prompt_improve_question = """Infer what a physician educator might want to know. This requires you to generate more specificity and then generate a greatly improved optimally effective question for submission to a GPT model.
For example, if the user asks "Tell me about indapamide" you respond, "Provide a comprehensive overview of indapamide, including its mechanism of action, indications, contraindications, common side effects, and important considerations for prescribing or monitoring patients?" 
Do not ask for more details - instead infer them and let the user update the details as needed, which they can do, before submitting the question to the GPT model. Solely return that updated question
with the improved specificity and detail optimized for direct answering by a GPT model.
"""

expert_instruction_content = """
The following is a conversation between a user and an assistant regarding a medical or scientific query. 
As an expert evidence-based academic physician and researcher, critically analyze the assistant's response. 
Please assess the response for scientific accuracy, logical consistency, and the presence of evidence-based reasoning. 
Be appropriately skeptical and identify any unsupported claims, logical fallacies, or deviations from established medical or scientific consensus. 
Your analysis should help ensure that the information provided is reliable, accurate, and in line with current best practices in evidence-based medicine and research.
Finally - provide a final list of any errors or unsupported claims in the response and a corrected response.
"""

teaching_styles = {
    "friendly_teacher": """
    Mimics the approach of a supportive and understanding educator who guides learners through the material in a conversational and engaging manner.
    This style emphasizes patience, encouragement, and personalized feedback, making learners feel valued and supported throughout their educational journey.
    The 'friendly teacher' adapts to the individual's pace and provides explanations, anecdotes, and examples in a way that is easy to understand and relate to.
    By fostering a positive and welcoming learning environment, this method aims to reduce anxiety and increase motivation, making learning a more enjoyable and less intimidating experience.
    Interactive elements such as quizzes, casual discussions, and reflective exercises are used to reinforce concepts and gauge understanding in a non-judgmental way.
    """,
    "socratic_method": """
    Encourages learners to think critically and articulate their thoughts through a series of questions and answers. 
    Ideal for stimulating deep understanding and reflection on complex medical cases or ethical dilemmas. 
    This method promotes active dialogue and debate, pushing users to explore and defend various viewpoints.
    """,
    "problem_based_learning": """
    Presents users with real-world medical problems to solve, fostering the application of theoretical knowledge to practical scenarios.
    This approach encourages self-directed learning, research, and collaboration among users to find solutions, simulating real-life medical decision-making processes.
    """,
    "flipped_classroom": """
    Inverts the traditional learning model by having users first explore content independently, such as reading articles or watching videos uploaded to the app.
    Subsequently, the chatbot engages them in interactive, application-focused activities based on this pre-learnt material, enhancing comprehension and retention.
    """,
    "microlearning": """
    Delivers content in small, manageable segments, focusing on a single concept or skill at a time.
    Ideal for busy medical professionals, this method facilitates quick learning sessions that fit into tight schedules, making it easier to absorb and retain information.
    """,
    "narrative_based_learning": """
    Uses storytelling and scenarios to convey complex information in a more relatable and engaging manner.
    By embedding medical knowledge within stories or patient cases, it aids in memorizing facts and understanding procedures, making learning more intuitive and less abstract.
    """,
    "simulation_and_role_play": """
    Allows users to engage in simulated clinical scenarios or role-play exercises, enhancing practical skills, decision-making, and interpersonal communication.
    This method provides a safe environment to practice and make mistakes, crucial for building confidence in clinical settings.
    """,
    "gamification": """
    Incorporates elements of game design, such as points, levels, and badges, to make the learning process more engaging and motivating.
    By setting challenges and rewards, it encourages continuous interaction and progress, making education feel more like play.
    """
}


ten_questions = """
Input: Outline of an Article 
Output: Return 5 questions (no answers) whose answers are *likely covered* in the full document. JSON format.

Objective: Return questions likely covered in the article that would be of interest to a physician.

Instructions:
1. Examine the outline to identify main subjects and details.
2. Craft straightforward, one specific aspect at at time questions to explore, e.g., diagnosis, treatment, prognosis, etc.

Output format: Return 5 questions, JSON format, without answers.

Example JSON formatting and type of questions:

```
{
  "Question1": "What are the pathophysiological mechanisms underlying Gastroesophageal Reflux Disease (GERD)?",
  "Question2": "How are the clinical manifestations of GERD different from those of Laryngopharyngeal Reflux (LPR)?",
  "Question3": "What evidence-based treatments are recommended for GERD?",
  "Question4": "What are the screening guidelines for Barrett's Esophagus in patients with GERD?",
  "Question5": "What are the indications for Endoscopic Eradication Therapy (EET) in the management of GERD?"
}
```

"""

