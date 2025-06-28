from dataflow.prompts.agenticrag import AtomicTaskGeneratorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow  import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import pandas as pd
import json

@OPERATOR_REGISTRY.register()
class AtomicTaskGenerator(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC = None
                 ):
        self.logger= get_logger()
        self.prompts = AtomicTaskGeneratorPrompt()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
            )
        elif lang == "en":
            return (
            )
        else:
            return
    
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, dataframe, prompt_type:str = None):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        if prompt_type == "get_conclusion":
            input_prompts = dataframe[self.input_key].tolist()
            system_prompts = self.prompts.initial_conclusion_system_prompt()
            prompts = [self.prompts.initial_conclusion_prompt(input_prompts) for input_prompts in input_prompts]
        elif prompt_type == "init_question":
            input_prompts = dataframe["candidate_tasks"].tolist()
            system_prompts = self.prompts.initial_question_system_prompt()
            prompts = [
                self.prompts.initial_question_prompt(item['conclusion'], item['R']) 
                for item in input_prompts
            ]
        elif prompt_type == "clean_qa":
            input_prompts = dataframe["generated_qa"].tolist()
            system_prompts= self.prompts.clean_qa_system_prompt()
            prompts = [
                self.prompts.clean_qa_prompt(item) 
                for item in input_prompts
            ]

        return system_prompts, prompts
    
    def _clean_json_block(self, item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def run(
            self,
            storage: DataFlowStorage,
            input_key:str = "prompts",
            output_key:str = "generated_task"
            ):
        '''
        
        '''
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        # step1: get conclusions
        formatted_system_prompts, formatted_prompts = self._reformat_prompt(dataframe, "get_conclusion")
        conclusion = self.llm_serving.generate_from_input(formatted_prompts, formatted_system_prompts)


        candidate_tasks = []

        for idx, (row, output_str) in enumerate(zip(dataframe.itertuples(index=False), conclusion)):
            try:
                parsed = json.loads(self._clean_json_block(output_str))
            except Exception as e:
                print(f"[WARN] JSON parse failed: {e} | output: {output_str}")
                parsed = []

            if not isinstance(parsed, list):
                parsed = []

            if len(parsed) == 0:
                continue
            else:
                for item in parsed:
                    if isinstance(item, dict) and "conclusion" in item and "R" in item:
                        new_row = row._asdict()
                        new_row["candidate_tasks"] = item  # 只添加一条任务
                        candidate_tasks.append(new_row)
        
        dataframe = pd.DataFrame(candidate_tasks)
        
        # step2: get atomic task question based on conclusions
        formatted_system_prompts, formatted_prompts = self._reformat_prompt(dataframe, "init_question")
        parsed_results  = self.llm_serving.generate_from_input(formatted_prompts, formatted_system_prompts)
        input_qas = []
        valid_indices = []

        for idx, (parsed_str, row) in enumerate(zip(parsed_results, dataframe.itertuples(index=False))):
            try:
                parsed = json.loads(parsed_str)
            except Exception as e:
                print(f"[WARN] Failed to parse JSON at index {idx}: {e} | value: {parsed_str}")
                input_qas.append(None)
                continue
            
            if not isinstance(parsed, dict) or "Q" not in parsed:
                input_qas.append(None)
                continue
            
            question = parsed["Q"]
            conclusion = row.candidate_tasks["conclusion"]

            input_qas.append({
                "question": question,
                "original_answer": conclusion,
            })
            valid_indices.append(idx)
        
        dataframe = dataframe.iloc[valid_indices].copy()
        dataframe["generated_qa"] = input_qas

        # step3: clean_qa
        formatted_system_prompts, formatted_prompts = self._reformat_prompt(dataframe, "clean_qa")
        cleaned_qa  = self.llm_serving.generate_from_input(formatted_prompts, formatted_system_prompts)
        new_data = []
        for res in cleaned_qa:
            try:
                res_dict = json.loads(res)
            except Exception as e:
                self.logger.warning(f"[WARN] Failed to parse cleaned_qa item: {e} | item: {res}")
                res_dict = {}

            new_data.append({
                "question": res_dict.get("question", None),
                "answer": res_dict.get("refined_answer", None),
                "identifier": "text"
            })
        dataframe["clean_qa"] = new_data
        
        # step4: filter qa
        # formatted_system_prompts, formatted_prompts = self._reformat_prompt(dataframe, "clean_qa")
        # cleaned_qa  = self.llm_serving.generate_from_input(formatted_prompts, formatted_system_prompts)

        # TODO: Search and Verify module

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return