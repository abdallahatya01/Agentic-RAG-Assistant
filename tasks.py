from crewai import Task

# 🧭 Router Task — يقرر هل نستخدم الـ Vectorstore أو Web Search
def create_router_task(router_agent):
    return Task(
        description=(
            "Analyze the given question {question} to determine the appropriate search method:\n\n"
            "1. Use 'vectorstore' if:\n"
            "   - The question relates to known topics or documents in the local RAG database.\n"
            "2. Use 'web_search' if:\n"
            "   - The topic requires current or real-time information.\n"
            "   - The question is about general or external knowledge.\n\n"
            "Make the decision based on meaning, not just keywords."
        ),
        expected_output=(
            "Return exactly one word:\n"
            "'vectorstore' — if the answer can be found in the local RAG database.\n"
            "'web_search' — if external data or current information is needed.\n"
            "No additional explanation or preamble."
        ),
        agent=router_agent,
    )

# 🔍 Retriever Task — يستخدم أداة حسب مخرجات الراوتر
def create_retriever_task(retriever_agent, router_task, rag_tool, web_search_tool):
    return Task(
        description=(
            "You have been given a routing decision from the Router agent: either 'vectorstore' or 'web_search'.\n"
            "1. If it's 'vectorstore', call rag_tool with the question: {question}.\n"
            "2. If it's 'web_search', call web_search_tool with the question: {question}.\n"
            "After you receive the tool output, you MUST immediately return it as your FINAL ANSWER.\n"
            "Do not think further. Do not use any tool again. Do not add commentary.\n"
            "Your only job is to pass through the raw tool output."
        ),
        expected_output=(
            "The EXACT raw output from the selected tool (rag_tool or web_search_tool), with no changes, "
            "no summary, and no extra text. Just the facts as returned by the tool."
        ),
        agent=retriever_agent,
        context=[router_task],
        tools=[rag_tool, web_search_tool],
    )

# 🧮 Grader Task — يقيم مدى الصلة بين السؤال والمحتوى
def create_grader_task(grader_agent, retriever_task):
    return Task(
        description=(
            "Evaluate whether the retrieved content from retriever_task is relevant to the question {question}."
        ),
        expected_output=(
            "Return only one word:\n"
            "'yes' — if the retrieved content answers or aligns with the question.\n"
            "'no' — if it is irrelevant or off-topic.\n"
            "No extra explanation."
        ),
        agent=grader_agent,
        context=[retriever_task],
    )

# 🤖 Hallucination Checker — يتحقق إن الإجابة مدعومة بمصادر فعلية
def create_hallucination_task(hallucination_grader, grader_task):
    return Task(
        description=(
            "Determine whether the response from retriever_task (evaluated by grader_task) is factually supported."
        ),
        expected_output=(
            "Return only one word:\n"
            "'yes' — if the answer is grounded in facts or contextually supported.\n"
            "'no' — if the answer is hallucinated or lacks factual grounding.\n"
            "No explanations."
        ),
        agent=hallucination_grader,
        context=[grader_task],
    )

# 🧩 Final Answer Task — ينتج النتيجة النهائية بناءً على نتيجة الـ hallucination
def create_answer_task(answer_grader, hallucination_task, retriever_task, web_search_tool):
    return Task(
        description=(
    "Based on the hallucination evaluation and the original retrieved content for question {question}:\n"
    "1. If hallucination check is 'yes':\n"
    "   → Extract only the part of the retrieved context that directly answers the question.\n"
    "   → Return it as a clear, standalone sentence.\n"
    "2. If hallucination check is 'no':\n"
    "   → Use web_search_tool to get a factual answer.\n"
    "3. Never add stories, examples, or external interpretation."
        ),
        expected_output=(
    "A concise, factual answer that directly addresses the user's question based solely on the retrieved context. "
    "Do not add external knowledge, assumptions, or examples. "
    "If the context does not contain the answer, do not fabricate one."
        ),
        agent=answer_grader,
        context=[hallucination_task, retriever_task],  # ← الإضافة الحاسمة
        tools=[web_search_tool],
    )