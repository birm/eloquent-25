import pandas as pd 
import ollama
from ollama import chat
from ollama import ChatResponse
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_parquet("Eloquent.parquet")
df['llm_quality_pref'] = ''
df['llm_quality_reasoning'] = ''

preamble = """Please tell me which of the two responses (A or B) a human is likely to prefer. Try to make a decisive decision, even if marginal. Please respond with simply A or B. When making your decision, please prioritize responses that are:"""

# removed both_good and both_bad as options for this analysis for now.

# initial
rules = """"
    - Directly relevant: Avoid including extraneous details; stay focused on answering the core of the user’s request.
    - Actionable and practical: Provide clear steps, advice, or tips that will help the user achieve their goals or understand the concept better.
    - Natural and approachable: Favor responses that feel conversational and are easy to relate to, avoiding overly formal or robotic language.
    - Well-structured and readable: Ensure the response is organized with clear paragraphs and formatting that makes it easy to read and follow.
    - Appropriately toned: Choose a tone that fits the context, professional when needed, or casual/friendly for informal situations, aligning with the intended audience.""" # 16/99

# iteration 2
rules = """
    - Depth and Completeness: Prefer responses that provide sufficient context, explanation, and supporting details over those that are overly brief. Avoid excessive conciseness when it leads to missing useful information.
    - Accuracy and Reliability: Prioritize responses that are factually correct and free from errors over those that may be concise but contain inaccuracies or misleading information. If one response includes external sources or references, ensure they are relevant and support the answer effectively.
    - Readability and Formatting: Favor responses that flow naturally and are easy to read. Well-structured formatting (e.g., spacing, paragraph breaks, numbered steps) improves comprehension.
    - Engagement and Tone: Prefer responses that feel natural and engaging rather than robotic, formulaic, or overly rigid. When applicable, warmth and personality can enhance user experience.
    - Relevance and Focus: Responses should stay on topic and directly answer the prompt, avoiding unnecessary tangents. However, additional background/context is beneficial when it enhances understanding rather than being excessive.
    - Tradition and Cultural Expectations: For topics with cultural or traditional importance (e.g., recipes, historical accounts), prefer responses that align with widely accepted norms unless explicitly asked for alternatives. 
""" # 51/99

# iteration 3

rules = """
    - **Naturalness and Human-like Flow:** Prefer responses that sound natural, fluid, and conversational rather than robotic or overly structured. Avoid excessive formality unless the context demands it.
    - **Clarity and Conciseness:** Favor responses that fully answer the prompt in a clear and concise manner. Avoid excessive detail that makes the response feel bloated or redundant.
    - **Accuracy and Relevance:** Ensure responses are factually correct and directly address the user's request. Avoid unnecessary tangents or unrelated details.
    - **Readability and Formatting:** Responses should be easy to read, using clear sentence structures, logical flow, and appropriate paragraph breaks. Avoid excessive formatting symbols that may make responses look unnatural.
    - **Engagement and Tone:** Prefer responses that are engaging, warm, and friendly when applicable. Avoid responses that feel dry, mechanical, or overly technical unless the prompt requires it.
    - **Prompt Alignment:** Responses should strictly adhere to the user's request. If the prompt asks for specific details, provide them concisely. If the prompt is broad, ensure the response remains focused without adding irrelevant personal interpretations.
    - **Factually Correct and Accessible:** Ensure that all provided information is accurate and well-structured. Prioritize responses that are informative but not overwhelming.
    - **Balanced Detail:** Responses should have enough depth to be informative but not so much that they become tedious or difficult to follow.
""" # 44/99

# iteration 4
rules = """
1. Naturalness and Human-like Flow:  
   Favor responses that sound natural, fluid, and conversational. Avoid responses that are robotic, overly structured, or formal unless the context demands it.

2. Clarity and Conciseness:  
   Prioritize responses that are clear and concise. Avoid unnecessary details that make the response feel bloated or redundant. However, ensure the response is sufficiently complete to fully address the prompt.

3. Factual Correctness and Depth:  
   Ensure responses are factually accurate and offer enough depth to answer the prompt thoroughly. When two responses are equally clear, the one with better accuracy and comprehensiveness should be preferred.

4. Engagement and Readability:  
   Responses should be engaging and easy to read. Favor responses with a warm, friendly tone (when appropriate) and a logical structure. Avoid responses that feel dry, mechanical, or overly technical unless required by the prompt.

5. Accuracy and Relevance:  
   Responses should be factually correct and directly answer the user's request. Avoid unnecessary tangents or unrelated details.

6. Balanced Detail:  
   Responses should strike a balance between depth and brevity. Provide enough detail to be informative without overwhelming or making the response difficult to follow.

7. Prompt Alignment:  
   Ensure that the response aligns with the user's request. If the prompt asks for specific details, provide them concisely. For broader prompts, keep the response focused and relevant without adding personal interpretations.
""" # 50/99

# iteration 5
rules = """
1. Naturalness and Human-like Flow:
   Favor responses that are natural, fluid, and conversational. Avoid responses that are robotic, overly structured, or formal unless the context demands it. Responses should feel approachable and engaging, akin to how a human would naturally communicate.

2. Clarity and Conciseness:
   Prioritize responses that are clear and concise. Avoid unnecessary details that make the response feel bloated or redundant. However, ensure the response is sufficiently complete to fully address the prompt, without oversimplifying or missing key points.

3. Factual Correctness and Depth:
   Ensure responses are factually accurate and offer enough depth to answer the prompt thoroughly. When two responses are equally clear, favor the one with better accuracy and comprehensiveness. But avoid making responses overly detailed if it detracts from clarity or readability.

4. Engagement and Readability:
   Responses should be engaging and easy to read. Favor responses with a warm, friendly tone (when appropriate) and a logical structure. Avoid responses that feel dry, mechanical, or overly technical unless required by the prompt. Keep responses inviting and relatable.

5. Balanced Detail and Brevity:
   Provide enough detail to be informative without overwhelming or making the response difficult to follow. Strike a balance between depth and brevity—responses should not be too long or include unnecessary tangents. When additional detail doesn't enhance clarity, it should be minimized.

6. Prompt Alignment:
   Ensure that the response aligns with the user's request. If the prompt asks for specific details, provide them concisely. For broader prompts, keep the response focused and relevant without adding personal interpretations or irrelevant details. Ensure all responses are aligned with the user's needs.

7. Inappropriate Length or Over-Detailed Responses:
   Responses should avoid excessive length or unnecessary detail unless it significantly contributes to the answer. When responses are verbose, prioritize the quality and relevance of the content over length. Avoid responses that might seem too academic or overly long for the context.
""" # score: 49/99

# iteration 6
rules = """
1. **Naturalness and Human-like Flow**:
   Prioritize responses that maintain a natural, conversational tone. Avoid mechanical, overly structured responses unless contextually necessary. Responses should feel human-like, engaging, and approachable, with a logical flow that mirrors how a person might naturally communicate.

2. **Clarity and Conciseness**:
   Ensure the response is clear and concise while maintaining the necessary level of detail to fully answer the prompt. Avoid over-explaining or using complex language that could confuse the reader. The response should be easily digestible, providing a straightforward answer without unnecessary elaboration.

3. **Engagement and Readability**:
   Responses should be engaging and easy to read, with an inviting tone. Strive for responses that maintain reader interest without being overly dry or mechanical. Where appropriate, use warm and friendly language. Responses should be relatable and feel human, balancing detail with an engaging presentation.

4. **Factual Correctness and Depth**:
   While it's important to keep responses clear and concise, they must also provide accurate and comprehensive information. Factual correctness is paramount. When two responses are equally clear, prioritize the one that provides more depth and accuracy without becoming overly verbose.

5. **Balanced Detail and Brevity**:
   Strike a balance between providing sufficient information and maintaining brevity. Responses should offer complete information while avoiding excessive length that would detract from clarity and ease of understanding. Focus on quality over quantity—brevity should never come at the expense of important details.

6. **Prompt Alignment**:
   Ensure the response fully aligns with the user's request. If the prompt asks for specific information, provide that directly, but without going off-topic. For broader prompts, keep the response focused on the most relevant aspects, avoiding unnecessary elaborations or irrelevant details.

7. **Improved Tone and Structure**:
   Use a tone that is conversational, informative, and approachable. Ensure the structure of the response is easy to follow, with clear transitions between ideas. Avoid robotic or overly formal language unless absolutely required by the context.

8. **Avoiding Overly Simplified or Unnatural Responses**:
   While brevity is important, responses should not be overly simplistic to the point of losing nuance. Aim for a balance where the response remains thorough and complete, yet still natural and engaging, rather than cutting out important context to maintain brevity.

9. **Improve Engagement by Relating to the User**:
   Engage the reader by offering helpful, real-world applications or relatable examples where possible. When appropriate, include details or side notes that make the content feel more personal, tailored, or approachable, especially for prompts with a creative or personal nature.

10. **Precision and Relevance**:
   Be precise with the information provided. When answering, focus on the most important points and avoid adding fluff or tangential details that don’t serve the prompt’s main purpose. Precision in word choice and staying relevant to the prompt is essential to maintain quality.
""" # 46/99

# iteration 7
rules= """
1. **Natural and Human-Like**: Prefer the response that feels more conversational, engaging, and natural to a human reader. Avoid overly technical, stiff, or robotic language.
2. **Relevant and Complete**: Choose the response that fully addresses the prompt's request. Avoid answers that stray off-topic or leave essential aspects unexplained.
3. **Concise and Clear**: Select the response that communicates the key points more succinctly, without unnecessary verbosity. Simplicity should not come at the cost of important information.
4. **Factually Accurate**: When both responses are equally clear, prioritize the one that offers more accurate and reliable information. Verify that all factual claims align with the prompt.
5. **Depth and Detail**: Favor the response that demonstrates deeper understanding or provides more comprehensive information, especially when it's presented in an organized, structured way.
6. **Readability and Flow**: Pick the response that flows better and is easier to read. This includes maintaining clear transitions, paragraph structure, and overall coherence.
7. **Engagement and Interest**: Choose the answer that feels more engaging or relatable. This could mean incorporating examples, stories, or details that make the response more captivating to read.

Please avoid bias towards responses that seem more formal or concise if they lose naturalness or engagement. Ensure the response selected provides the most helpful and human-like explanation of the content.
""" # 56/99

# switched to deep seek for fun:
rules = """
1. Prioritize Natural Conversational Flow Above All:
   - Choose responses that use everyday language, contractions, and natural phrasing
   - Avoid rigid structures, bullet points, or markdown symbols (*, -)
   - Prefer responses that "show" rather than "tell" through examples

2. Value Conciseness Over Comprehensive Detail:
   - Favor focused answers that directly address the prompt
   - Penalize unnecessary technical jargon or redundant explanations
   - Accept slightly shorter responses if they maintain clarity

3. Verify Factual Accuracy First:
   - Reject responses with any factual errors immediately
   - Prioritize correctness over stylistic polish
   - When technical precision matters, prefer specific numbers/dates over vague terms

4. Preserve Contextual Relevance:
   - Eliminate responses that add unrelated information
   - Prefer answers that maintain consistent perspective/voice
   - Penalize "creative" interpretations that stray from prompt requirements

5. Optimize for Practical Implementation:
   - Choose actionable advice over theoretical frameworks
   - Prefer numbered steps only when sequence matters
   - Favor responses with real-world examples/caveats

6. Format for Human Readability:
   - Avoid robotic templating (e.g., "In conclusion...")
   - Prefer paragraph breaks over bullet points
   - Allow occasional sentence fragments for conversational effect

7. Surface-Level Engagement Checks:
   - Penalize academic/lecturing tones
   - Reward subtle humor/wordplay when appropriate
   - Prefer active voice and varied sentence lengths
""" # 51/99

rules = """
1. Factual Gatekeeping (Non-Negotiable)
   - Reject ANY response with factual errors or hallucinations first
   - When both are accurate, proceed to next criteria

2. Strict Prompt Adherence
   - Eliminate responses adding unsolicited information ("extra tips")
   - Prefer responses mirroring prompt's exact focus/scope
   - Penalize creative expansions beyond original request

3. Anti-Robot Formatting
   - Ban ALL markdown symbols (*, -, •) - instant rejection
   - Prefer paragraph breaks over lists (even if less organized)
   - Allow occasional sentence fragments for natural flow

4. Conversational Priority
   - Require contractions (it's, don't) and casual phrasings
   - Prefer "you"/"we" over third-person constructions
   - Reward subtle humor/wordplay when context-appropriate

5. Detail Threshold Balancing
   - Accept longer responses ONLY if:
     - Every detail directly serves prompt
     - No redundant adjectives/adverbs
     - Maintains natural speech rhythm

6. Implementation Focus
   - Choose actionable steps over theoretical frameworks
   - Prefer concrete examples over abstract concepts
   - Value safety disclaimers/real-world caveats

7. Mistake Amplification
   - Penalize these 2x severity:
     - Unprompted bullet points/lists
     - Technical jargon without explanation
     - Passive voice clusters (>1 per paragraph)
""" # 57/99

rules = """
1. **Human-Likeness Hierarchy** (Non-Negotiable Order):
   a) Natural conversational flow (contractions, colloquial phrasing)
   b) Contextual completeness (answers "why", not just "what")
   c) Factual accuracy
   d) Structural clarity

2. **Formatting Forgiveness**:
   - Ignore formatting issues if content is superior
   - Allow markdown/bullets ONLY when improving scannability
   - Prefer paragraph breaks but don't penalize lists for complex steps

3. **Anti-Robotic Speech Filters**:
   █ REJECT responses containing:
     - "It's important to note"
     - "Key considerations include"
     - "This [noun] demonstrates"
   █ PENALIZE 2x:
     - Passive voice clusters (>1 per 3 sentences)
     - Academic hedging ("may", "could", "potentially")

4. **Detail Sweet Spot**:
   Preferred Response = (Prompt Complexity × 1.5) + 2 Examples
   - Simple prompts: 1-3 concise points + 1 analogy
   - Complex prompts: 3-5 points + 2 real-world examples

5. **Engagement Multipliers**:
   Reward these 3x more than structure:
   - First-person anecdotes
   - Rhetorical questions
   - Situational humor (when context-appropriate)
   - Emotive descriptors ("frustrating", "exciting")

6. **Error Criticality Scale**:
   - Factual error = Instant rejection
   - Missing safety note = -5 pts
   - Formatting issue = -1 pt
   - Slight unnaturalness = -0.5 pts
""" # 55/99

rules = """
1. **Natural Conversational Primacy**
   - **Immediate Rejection Triggers:**
     - Robotic phrases ("It is important to note", "Key considerations")
     - Passive voice clusters (>1 instance per 3 sentences)
   - **Mandatory Elements:**
     - Contractions ("you'll", "they're")
     - Direct address ("you", "we") where appropriate
     - Conversational rhythm (mix of short/long sentences)

2. **Precision-Trimmed Detail**
   - Allow details ONLY if:
     1. Directly answers explicit prompt elements
     2. Presented as organic story elements (not bullet lists)
     3. Contains zero redundant adjectives/adverbs
   - For technical topics:
     - 1 analogy + 2 examples max
     - Must explain jargon in context

3. **Contextual Completeness Scale**
   | Prompt Type          | Required Elements                          | Detail Allowance   |
   |----------------------|--------------------------------------------|--------------------|
   | Creative/Narrative   | Character perspective, sensory details    | High (metaphors OK)|
   | Technical/How-To     | Step context ("why" before "how")          | Medium (+1 safety)|
   | Factual Explanation  | Single concrete example + real-world impact| Low (strict focus) |

4. **Safety/Implementation Lock**
   - Automatic preference to responses containing:
     - Safety caveats (recipes/health/legal)
     - "Check local..." advisories
     - Time/cost estimates (when relevant)
   - Even if slightly less natural

5. **Anti-Overengineering Filters**
   - Penalize 2x for:
     - Unprompted bullet points/markdown
     - Triple adjectives ("quick, easy, delicious")
     - Hedging verbs ("may", "could", "might")
   - Reward 3x for:
     - Single perfect analogy
     - Authentic personal voice markers
     - Anticipated follow-up questions

6. **Error Criticality Hierarchy**
   1. Factual inaccuracy → Instant rejection
   2. Missing safety note → -5 pts
   3. Unnatural phrasing → -2 pts
   4. Overly technical → -1 pt
   5. Formatting issues → -0.5 pts

7. **Tiebreaker Protocol**
   A vs B when equal:
   1. Which sounds like something a friend would say?
   2. Which better anticipates likely follow-up questions?
   3. Which uses fewer syllables per idea?
""" # 54/99

for idx, row in df.iterrows():
    conversation_history = []
    message = f"\n\n{preamble}{rules}\nInstruction: {row['instruction']}\n\nResponse A: {row['output_a']}\nResponse B: {row['output_b']}\n\n\n"
    conversation_history.append({
        'role': 'user',
        'content': message,
    })
    
    valid_input = False
    chat_response: ChatResponse = chat(model='llama3.2', messages=conversation_history)
    chat_choice = chat_response.message.content.upper().strip()
    #print(chat_response.message.content)
    while not valid_input:
        # remove . which sometimes also shows up
        chat_choice = chat_choice.replace(".", "")
        #print(chat_choice)
        # is the model response already valid?
        quality_pref = chat_choice
        if chat_choice in ['A', 'B', 'BOTH_GOOD', 'BOTH_BAD']:
            df.at[idx, 'llm_quality_pref'] = quality_pref
            valid_input = True
        # starts with correct response case
        elif chat_choice.replace("\n", " ").split(" ")[0] in ['A', 'B', 'BOTH_GOOD', 'BOTH_BAD']:
            df.at[idx, 'llm_quality_pref'] = quality_pref
            valid_input = True
        else:
            print("chat said: ", chat_choice)
            user_input = input("Interpret model's judgement (A, B, -, or +): ").strip().upper()
            if user_input in ['A', 'B', '-', '+']:
                if user_input == "-":
                    user_input = "BOTH_BAD"
                if user_input == "+":
                    user_input = "BOTH_GOOD"
                quality_pref = user_input
                df.at[idx, 'llm_quality_pref'] = user_input
                valid_input = True
            else:
                print("Invalid input. Please enter A, B, -, or +.")
        conversation_history.append({
            'role': 'assistant',
            'content': quality_pref,
        })
        conversation_history.append({
            'role': 'user',
            'content': f"Please provide a brief, one-sentence explanation of why you chose {quality_pref}",
        })
        reasoning = chat(model='llama3.2', messages=conversation_history)
        df.at[idx, 'llm_quality_reasoning'] = reasoning.message.content

df.to_parquet("llama_32_gpt_prompt_distilation.parquet")
df['overall_quality_preference'] = df.overall_quality_preference.str.upper()

"""
## agreement contingency plot
zs_crosstab = pd.crosstab(df['llm_quality_pref'], df['overall_quality_preference'])
sns.heatmap(zs_crosstab, annot=True, cmap='YlGnBu', fmt='d').set_title('Zero Shot LLM vs Baseline Overall Quality Preference')
plt.tight_layout()
plt.show()
"""


score = sum(df.overall_quality_preference == df.llm_quality_pref)
print(f"score: {score}/{len(df)}")
print("---")

bad_ones = df[
    ((df.overall_quality_preference == "A") & (df.llm_quality_pref == "B")) |
    ((df.overall_quality_preference == "B") & (df.llm_quality_pref == "A"))
]

for idx, row in bad_ones.iterrows():
    print(f"The model preferred {row.llm_quality_pref} because '{row.llm_quality_reasoning}', "
          f"but the human preference was {row.overall_quality_preference} because '{row.overall_quality_explanation}'")

print("---")
print("Here's the old rules: ", rules)
