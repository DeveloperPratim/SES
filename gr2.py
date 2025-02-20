# coding: utf-8
import gradio as gr
import asyncio

# Async wrapper to handle the evaluation
async def evaluate_async(question, student_answer, expected_answer, full_marks):
    result = await GetData(question, student_answer, expected_answer, full_marks)
    return result

# Wrapper for Gradio to handle async
def evaluate(question, student_answer, expected_answer, full_marks):
    return asyncio.run(evaluate_async(question, student_answer, expected_answer, full_marks))

# Gradio interface
demo = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.Textbox(label="Question", value="Explain the concept of deadlock in operating systems."),
        gr.Textbox(label="Student Answer", value="Deadlock occurs when processes wait forever for resources held by each other."),
        gr.Textbox(label="Expected Answer", value="A deadlock is a situation where a set of processes are blocked because each process is holding a resource and waiting for another."),
        gr.Number(label="Full Marks", value=5)
    ],
    outputs=gr.Textbox(label="Evaluation Result"),
    title="Student Answer Evaluation",
    description="Enter the question, student answer, expected answer, and full marks to evaluate."
)

demo.launch()
