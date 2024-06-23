# summary_bot.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import intel_extension_for_pytorch as ipex

class SummaryBot:
    def __init__(self, model_name='sshleifer/distilbart-cnn-12-6', device='cpu'):
        print("Initializing the model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        if device.startswith('xpu'):
            self.model = ipex.optimize(self.model)

    def summarize(self, text, max_length=130, min_length=30):
        print("Generating summary...")
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

def main():
    device = 'cpu'  # Change to 'xpu' if you are using Xeon processors with Intel's optimization
    with open("raw-note.txt", "r", encoding="utf-8") as file:
        text = file.read()
    bot = SummaryBot(device=device)
    summary = bot.summarize(text)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
