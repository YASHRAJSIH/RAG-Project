import gradio as gr

# Placeholder function to simulate the RAG model response
def rag_response(input_text):
    # This would be the actual call to your RAG model.
    placeholder_output = "This is a sample response from the RAG model based on your input."
    return placeholder_output

# Define the Gradio interface
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        "<h1 style='text-align: center; color: white;'>RAG Model Chat Interface</h1>"
    )

    # Arrange input and output areas in a single vertically aligned column
    with gr.Column():
        prompt = gr.Textbox(
            placeholder="Type your prompt here...",
            label="Input",
            lines=2,
            max_lines=2,
            show_label=True,
            elem_classes="carved-border"  # Custom class for carved borders
        )
        response = gr.Textbox(
            label="RAG Model Output",
            placeholder="Model response will appear here...",
            lines=10,
            interactive=False,
            elem_classes="carved-border"  # Custom class for carved borders
        )

        # Buttons for submitting and stopping the response
        with gr.Row():
            submit_btn = gr.Button("Submit")
            stop_btn = gr.Button("Stop")

    # Link the Submit button to trigger the RAG model response function
    submit_btn.click(fn=rag_response, inputs=prompt, outputs=response)

# Add custom CSS for carved borders
demo.css = """
    .carved-border {
        border-radius: 8px;
        border: 2px solid grey;
        padding: 10px;
        background-color: #333333;
        color: white;
    }
    button {
        color: white;
        background-color: #555555;
        border-radius: 5px;
        border: none;
        padding: 8px 12px;
    }
    button:hover {
        background-color: #777777;
    }
"""

# Launch the Gradio app
demo.launch()
