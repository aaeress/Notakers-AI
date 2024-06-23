import torch
import os
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import intel_extension_for_pytorch as ipex

class ChatBotModel:
    """
    ChatBotModel is a class for generating responses based on text prompts using a pretrained model with IPEX optimization.
    """

    def __init__(
        self,
        model_id_or_path: str = "gpt2",
        torch_dtype: torch.dtype = torch.float32,
        optimize: bool = True,
    ) -> None:
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
        self.model_id_or_path = model_id_or_path

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_id_or_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_id_or_path).to(self.device).eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        if optimize:
            self.model = ipex.optimize(self.model, dtype=torch_dtype)

    def generate_output(self, text: str):
        """
        Generate structured output text based on the input text prompt.
        """
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_length=1024)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def format_structured_output(self, text):
        """
        Formats the generated text to have a structured layout with titles, sections, or bullet points.
        """
        lines = text.split('.')
        structured_text = "Your Great Note:\n\n"
        for line in lines:
            if "section:" in line:
                structured_text += f"## {line.replace('section:', '').strip()}\n\n"
            else:
                structured_text += f"- {line.strip()}\n"
        return structured_text

def main():
    model_path = "gpt2"  # GPT-2 model identifier
    chat_model = ChatBotModel(model_id_or_path=model_path, optimize=True)
    input_text = """
    Inspiration
High cost of learning is a common issue with many feature-rich note-taking software today. They may be excellent tools for some, but you might struggle to use them effectively. Cluttered notes make it challenging to revisit and relearn important information. Time is always being wasted when you try to organize and format notes manually. Meaningless, repetitive work to record similar ideas.

All these issues make your learning process cumbersome and inefficient. We aim not only to simplify these learning processes but also to create a software that integrates your knowledge and inspirations.

What it does
How we built it
Challenges we ran into
Accomplishments that we're proud of
What we learned
What's next for Notakers AI
Next, we plan to design a comprehensive knowledge base that seamlessly integrates all your notes, allowing users to draw inspiration and ideas from the interconnected content. This knowledge base will help users better organize and relate their thoughts and information, enhancing learning efficiency and fostering creative thinking.

We also aim to incorporate speech-to-text and text-to-speech functionalities. These features will be particularly beneficial for individuals with reading disorders and ADHD, enabling them to quickly and easily record and review their ideas. With speech-to-text, users can effortlessly convert spoken words into written notes, while text-to-speech allows them to listen to their notes, making comprehension and retention easier.

These enhancements not only simplify the learning process but also ensure that every user, regardless of their challenges, can efficiently capture and integrate their knowledge and inspiration. We believe that with these improvements, Notakers AI will become an indispensable tool for learning and creativity.
    """
    raw_output = chat_model.generate_output(input_text)
    formatted_output = chat_model.format_structured_output(raw_output)
    print(formatted_output)

if __name__ == "__main__":
    main()
