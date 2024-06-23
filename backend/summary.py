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
    text = """## Inspiration
High cost of learning is a common issue with many feature-rich note-taking software today. They may be excellent tools for some, but you might struggle to use them effectively. Cluttered notes make it challenging to revisit and relearn important information. Time is always being wasted when you try to organize and format notes manually. Meaningless, repetitive work to record similar ideas. 
All these issues make your learning process cumbersome and inefficient. We aim not only to simplify these learning processes but also to create a software that integrates your knowledge and inspirations.
## What it does

## How we built it

## Challenges we ran into

## Accomplishments that we're proud of

## What we learned

## What's next for Notakers AI
Next, we plan to design a comprehensive knowledge base that seamlessly integrates all your notes, allowing users to draw inspiration and ideas from the interconnected content. This knowledge base will help users better organize and relate their thoughts and information, enhancing learning efficiency and fostering creative thinking.
We also aim to incorporate speech-to-text and text-to-speech functionalities. These features will be particularly beneficial for individuals with reading disorders and ADHD, enabling them to quickly and easily record and review their ideas. With speech-to-text, users can effortlessly convert spoken words into written notes, while text-to-speech allows them to listen to their notes, making comprehension and retention easier.
These enhancements not only simplify the learning process but also ensure that every user, regardless of their challenges, can efficiently capture and integrate their knowledge and inspiration. We believe that with these improvements, Notakers AI will become an indispensable tool for learning and creativity."""
    bot = SummaryBot(device=device)
    summary = bot.summarize(text)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
