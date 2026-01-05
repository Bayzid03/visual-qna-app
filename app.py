import gradio as gr
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def answer_question(image, text):
    try:
        encoding = processor(image, text, return_tensors="pt")

        outputs = model(**encoding)
        logits = outputs.logits

        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        return answer
        
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I had trouble processing that. Try a different image or question."

# 6. Create the Gradio web interface
iface = gr.Interface(
    fn=answer_question, 
    inputs=[
        gr.Image(type="pil"), 
        gr.Textbox(label="Ask a question about the image...") 
    ],
    outputs=gr.Textbox(label="Answer"), 
    title="🤖 Multimodal AI: Visual Question Answering",
    description="Upload an image and ask any question about it. (Model: dandelin/vilt-b32-finetuned-vqa)"
)

# 7. Launch the app!
iface.launch()
