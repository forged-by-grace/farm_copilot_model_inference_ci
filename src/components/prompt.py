from src.entity.config_entity import PromptConfig
from openai import OpenAI
from typing import Dict


class Prompt:
    def __init__(self, question: str, config: PromptConfig = None, llm: OpenAI = None) -> None:
        self.question = question
        self.llm = llm
        self.config = config


    async def request_response(self):        
        response = self.llm.chat.completions.create(
            temperature=self.config.params_app_prompt_llm_temperature,
            model=self.config.params_app_prompt_model,
            messages=[{'role': self.config.params_app_prompt_role, 'content': self.question}],
        )

        self.response = response.choices[0].message.content


    def get_response(self) -> Dict[str, str]:
        return {'response': self.response}