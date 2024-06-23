# chatbot_model.py

import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import intel_extension_for_pytorch as ipex

class NoteModel:
    """
    NoteModel is a class for generating responses based on text prompts using a pretrained model with IPEX optimization.
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
        structured_text = "标题: 自动概述\n\n"
        for line in lines:
            if "section:" in line:
                structured_text += f"## {line.replace('section:', '').strip()}\n\n"
            else:
                structured_text += f"- {line.strip()}\n"
        return structured_text
