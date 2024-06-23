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
    Because people are still coming in, I think we will not start on until a few minutes. Can people at the back hear me? Let me see how I can do that. Let me see. Okay. I turn up the volume. Can you at the back hear me? Yes, I know. I yes, raise your hand. Can you at the back hear me? Okay, cool. And also, I want to verify that if I'm writing on the margin of the board, can it? So if I'm writing like P Q? Can I see this just fine? No. N. Okay. I'll try not to write on the board. Okay, let's start. So hi everyone. I'm Hong W. And I'll be teaching the first two weeks of the course and also some two weeks in the middle. And today, the lecture will be about mathematical proofs. And I put this in my lecture, I'll be primarily using slides because we have a relatively large class, and I think using slides, people at the back can see me more clearly. And also another reason is I don't want to torture you with my sometime terrible handwriting. I put this famous line by Fermat on the first page of my slide. Fermat say I have discovered the truly marvelous proof of this of this som, which, however, the margin of the paper is not large enough to contain. This is like a typical zero score answer on CS 70. But don't do this. But the story behind it is quite intriguing. So Fermat's last theorem was first conjectured, of course, by Fermat 400 years ago. But it took people really like something like 350 years to discover the final proof. Finally, a mathematician called Andrew Wells prove it. It's really amazing how mankind has spent 300 years just for a single mathematical proof of a simple fact. So that's why I think this lecture is particularly important because finding proof is really the heart of the pursuit of many mathematicians. Okay. Before we jump into this lecture, let's recap what we learned last time. And at any point, if you have questions, just raise your hand, if I see a few hands I'll pause and answer each one of the questions. Last time we learned propositions and more broadly proposition logic. So Positional logic is like the language of mathematics. Is the thing that you use to describe any mathematical statement, and propositions are like sentences in this language. For example, square root three is irrational. This is provocation, and 1.1 equals five, it's also provocation, it's wrong, but it's still wrong propsation. And so what is not the proposition? For example, two plus two, it's an expression, but it doesn't have a truth value to it. It didn't make any assertion for what two plus two should be equal to. So that's not the proposition. And also, if I just write three x equals six, that's an equation about x. But I haven't specified what x is. So x here is what called a free variable. It's free, can be anything. I haven't specify it. So in this case, three x equals six, it doesn't have a truth value. So it's not a proposition. Okay. And we also have variables to represent the propositions. For example, for this proposition, square root three is irrational. You can de no it by P. The P is nothing but a shorthand for this long sentence, which we don't want to write. Okay. Let's say we can also do this. And we also learn the connectives for propositional logic. These are the words that connect these sentences to form even longer sentences. For example, we learn this conjunction. This is in English, this is. The conjunction of P and Q, it is true only when P is true and Q is true. So it's literally just P and Q. And we learn disjunction, which is in English, it's equivalent to R. P or Q means it is true if at least one of P and Q is true. But note here, it is true when at least one of P and Q is true, meaning that when P and Q are both true, P Q is still true. This is a bit different from the normal use in English. Normally when will say either A or B, is sort of mean that only one of A and B is true, but that in mathematical logic is called exclusive or x or. For disjunction, it's okay that if P and K are both true, then this proposition is still true. 
    """
    raw_output = chat_model.generate_output(input_text)
    formatted_output = chat_model.format_structured_output(raw_output)
    print(formatted_output)

if __name__ == "__main__":
    main()
