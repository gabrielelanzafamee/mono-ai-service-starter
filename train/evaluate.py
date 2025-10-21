import os
import json
import weave

from openai import AsyncOpenAI

from llm.gemma import Gemma
from preprocess import Dataset
from settings import Config

class EvaluatorModel(weave.Model):
    prompt_template: str
    llm: Gemma

    @weave.op()
    async def predict(self, question: str) -> str:
        response = self.llm.generate([
            { "role": "system", "content": [{ "type": "text", "text": self.prompt_template }] },
            { "role": "user", "content": [{ "type": "text", "text": question }] }
        ])
        
        if response is None:
            raise ValueError("Response is None")
        
        return response

class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.run_name = f"{self.config.lora_id}_{self.config.dataset_id}_{self.config.model_id}".replace("/", "-")
        self.llm = Gemma(self.config)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    @weave.op()
    async def build_evaluation_dataset(self):
        self.dataset = Dataset(self.config.dataset_id)
        examples = []
        for item in self.dataset.test:
            example = { "question": "", "answer": "" }
            for message in item["messages"]:
                if message["role"] == "user":
                    example["question"] = message["content"][0]["text"]
                elif message["role"] == "assistant":
                    example["answer"] = message["content"][0]["text"]
            examples.append(example)
        return examples[:10]
    
    @weave.op()
    async def evaluate_scorer(self, answer: str, output: str):
        """
        Evaluate the model output against the ground truth answer.
        
        Args:
            answer: The ground truth answer from the dataset
            output: The model's predicted output
            
        Returns:
            bool: True if the output matches the answer exactly
        """
        prompt_template = f"You are a helpful assistant and evaluate the following original answer and the following generated output. Return a JSON object with the fields 'correct' that is a boolean and can be true or false, another field is accuracy, give a vote to 0 to 100 to the accuracy of the output based on the correct answer."
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                { "role": "system", "content": prompt_template },
                { "role": "user", "content": f"Original answer: {answer}\nGenerated output: {output}" }
            ],
            response_format={"type": "json_object"}
        )
        
        if response.choices[0].message.content is None:
            raise ValueError("Response is None")
        
        print("Evaluation response:")
        print(response.choices[0].message.content)
        
        return json.loads(response.choices[0].message.content)
    
    async def run(self):
        weave.init(self.config.project_name)
        
        evaluation_dataset = await self.build_evaluation_dataset()
        
        async def scorer_wrapper(answer, output):
            return await self.evaluate_scorer(answer, output)
        
        evaluation = weave.Evaluation(
            name=f"{self.config.lora_id}_{self.config.dataset_id}_{self.config.model_id}_evaluation",
            description="Evaluate the model on the evaluation dataset",
            dataset=evaluation_dataset,
            scorers=[scorer_wrapper]
        )
        
        self.llm.load_model(quantization=False)
        self.llm.load_lora(self.config.lora_id, from_hub=True)
        
        model = EvaluatorModel(
            prompt_template="You are a helpful assistant that can answer and understand Italian.",
            llm=self.llm
        )
        
        result = await evaluation.evaluate(model)
        return result