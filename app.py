import gradio as gr
from crewai import Crew
import time

# Import your existing modules
from tools import rag_tool, web_search_tool  
from agents import (
    create_router_agent, 
    create_retriever_agent, 
    create_grader_agent, 
    create_hallucination_grader, 
    create_answer_grader,
)
from tasks import (
    create_router_task, 
    create_retriever_task, 
    create_grader_task, 
    create_hallucination_task, 
    create_answer_task,
)


router_agent = create_router_agent()
retriever_agent = create_retriever_agent()
grader_agent = create_grader_agent()
hallucination_grader = create_hallucination_grader()
answer_grader = create_answer_grader()

router_task = create_router_task(router_agent)
retriever_task = create_retriever_task(retriever_agent, router_task, rag_tool, web_search_tool)
grader_task = create_grader_task(grader_agent, retriever_task)
hallucination_task = create_hallucination_task(hallucination_grader, grader_task)
answer_task = create_answer_task(answer_grader, hallucination_task, retriever_task, web_search_tool)

rag_crew = Crew(
    agents=[router_agent, retriever_agent, grader_agent, hallucination_grader, answer_grader],
    tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
    verbose=True,
)


def ask_question(user_question):
    start = time.time()
    result = rag_crew.kickoff(inputs={"question": user_question})
    end = time.time()
    return f"⏱ Time: {end-start:.2f} sec\n\n📌 Answer:\n{result}"


interface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(
        label="Ask anything",
        placeholder="مثال: What is the Self-Attention Mechanism in Transformers?",
        lines=2
    ),
    outputs=gr.Textbox(
        label="Chatbot Response",
        lines=20,           
        max_lines=40,
        show_copy_button=True,
    ),
    title="Agentic RAG Chatbot",
    description="A smart multi-agent RAG system using CrewAI."
)

if __name__ == "__main__":
    interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    inbrowser=True
)



