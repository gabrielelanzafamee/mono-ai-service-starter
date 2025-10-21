import torch

from datasets import load_dataset

class Dataset:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id

        self.dataset = load_dataset(dataset_id)
        self.split = self.dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
        self.train = self.split["train"]
        self.test = self.split["test"]
        
        # shuffle data
        self.train = self.train.shuffle(seed=42)
        self.test = self.test.shuffle(seed=42)

        self.train = self.train.map(self._format_instruction)
        self.test = self.test.map(self._format_instruction)

        self.train = self.train.remove_columns(["conversation"]) # keep only required columns
        self.test = self.test.remove_columns(["conversation"]) # keep only required columns
        
    def _format_instruction(self, dataset_item):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that can answer and understand Italian."}]},
        ]
        
        for item in dataset_item["conversation"]:
            messages.append({ "role": item.get("role"), "content": [{"type": "text", "text": item.get("content")}]})
        
        return {
            "messages": messages
        }
        
    def _preprocess_messages(self, messages, llm_instance):
        inputs = llm_instance.processor.apply_chat_template(
            messages,
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(llm_instance.model.device, dtype=llm_instance.model.dtype)
        input_ids = inputs['input_ids'][0]
        return input_ids
        
    def analyse_dataset(self, llm_instance):
        train_tokenized_lenghts = []
        test_tokenized_lenghts = []
        
        for item in self.train:
            train_tokenized_lenghts.append(len(self._preprocess_messages(item["messages"], llm_instance)))
        
        for item in self.test:
            test_tokenized_lenghts.append(len(self._preprocess_messages(item["messages"], llm_instance)))
            
        train_max_lenght = max(train_tokenized_lenghts)
        test_max_lenght = max(test_tokenized_lenghts)
        
        print(f"Train max lenght: {train_max_lenght}")
        print(f"Test max lenght: {test_max_lenght}")
        
        return train_max_lenght, test_max_lenght