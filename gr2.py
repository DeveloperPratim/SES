
import gradio as gr
import asyncio
import time

# Async wrapper to handle the evaluation
async def evaluate_async(question, student_answer, expected_answer, full_marks, progress=gr.Progress()):
    start_time = time.time()
    for i in range(1, 6):
        progress((i * 20) / 100, desc=f"⏳ Processing... {i * 20}%")
        time.sleep(0.3)  # Simulate processing steps

    result = await GetData(question, student_answer, expected_answer, full_marks)
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    return result, elapsed_time

# Wrapper for Gradio to handle async, show progress, and time
def evaluate(question, student_answer, expected_answer, full_marks, progress=gr.Progress()):
    progress(0, desc="🚀 Starting evaluation...")

    result, elapsed_time = asyncio.run(evaluate_async(question, student_answer, expected_answer, full_marks, progress))

    final_marks = result.get('final_marks_obtained', 'N/A')
    feedback = result.get('feedback', 'No feedback provided.')
    evaluation_text = f"### 📝 Evaluation Feedback\n{feedback}\n\n---\n\n```json\n{result}\n```"
    final_marks_text = f"🏆 **Final Obtained Marks:** {final_marks} / {full_marks}"
    time_text = f"⏱️ **Time Taken:** {elapsed_time} seconds"

    return evaluation_text, final_marks_text, time_text

# Improved Gradio interface with progress bar and time display
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # 📖 Student Answer Evaluation System
        **Evaluate student responses with AI-powered grading.**
        *(See live processing progress and time taken.)*
        """
    )

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="📝 Question",
                lines=3,
                placeholder="Enter the question here...",
                value="Explain the concept of deadlock in operating systems."
            )
            student_answer = gr.Textbox(
                label="✍️ Student Answer",
                lines=5,
                placeholder="Enter the student's answer...",
                value="Deadlock occurs when processes wait forever for resources held by each other."
            )
            expected_answer = gr.Textbox(
                label="✅ Expected Answer",
                lines=5,
                placeholder="Enter the ideal answer...",
                value="A deadlock is a situation where a set of processes are blocked because each process is holding a resource and waiting for another."
            )
            full_marks = gr.Number(label="🏆 Full Marks", value=5, interactive=True)
            submit_btn = gr.Button("🔍 Evaluate")

        with gr.Column():
            evaluation_result = gr.Markdown(label="📊 Evaluation Result")
            final_marks_display = gr.Markdown(label="🥇 Final Obtained Marks")
            time_display = gr.Markdown(label="⏱️ Processing Time")

    submit_btn.click(
        evaluate,
        inputs=[question, student_answer, expected_answer, full_marks],
        outputs=[evaluation_result, final_marks_display, time_display],
        show_progress=True
    )

demo.launch()
