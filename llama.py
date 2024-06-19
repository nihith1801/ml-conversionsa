from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.agents import Agent
from langchain.prompts import PromptTemplate

# Define the pre-prompt
pre_prompt = """
Your name is Rella, and you are a multispeciality doctor
"""

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Initialize the pipeline with the pre-prompt
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # Using CPU, device=-1

# Define the LangChain agent
class MyAgent(Agent):
    def __init__(self, pipeline, pre_prompt):
        self.pipeline = pipeline
        self.pre_prompt = pre_prompt

    def run(self, prompt):
        full_prompt = self.pre_prompt + "\n" + prompt
        response = self.pipeline(full_prompt, max_length=512, num_return_sequences=1)
        return response[0]['generated_text'][len(full_prompt):]

# Instantiate the agent
agent = MyAgent(pipe, pre_prompt)

# Example usage
query = "What are the benefits of using renewable energy sources?"
response = agent.run(query)
print(response)
