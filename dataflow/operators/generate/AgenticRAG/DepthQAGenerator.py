from dataflow.prompts.agenticrag import DepthQAGeneratorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow  import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import pandas as pd
import json

@OPERATOR_REGISTRY.register()
class DepthQAGenerator(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC = None
                 ):
        self.logger= get_logger()
        self.prompts = DepthQAGeneratorPrompt()
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
        if prompt_type == "get_indentifier": ### havent use yet
            input_prompts = dataframe[self.input_key].tolist()
            system_prompts = self.prompts.get_identifier_system_prompt()
            prompts = [self.prompts.get_identifier_prompt(input_prompts) for input_prompts in input_prompts]
        elif prompt_type == "get_backward":
            input_prompts = dataframe["identifier"].tolist()
            system_prompts = ""
            prompts = [self.prompts.get_backward_task_prompt(input_prompts) for input_prompts in input_prompts]
        elif prompt_type == "check_superset":
            new_identifiers = dataframe["new_identifier"].tolist()
            relations = dataframe["relation"].tolist()
            identifiers = dataframe["identifier"].tolist()
            system_prompts = self.prompts.check_superset_system_prompt()
            prompts = [self.prompts.check_superset_prompt(new_id, relation, identifier) for new_id, relation, identifier in zip(new_identifiers, relations, identifiers)]
        elif prompt_type == "get_new_question":
            new_identifiers = dataframe["new_identifier"].tolist()
            relations = dataframe["relation"].tolist()
            identifiers = dataframe["identifier"].tolist()
            system_prompts = self.prompts.get_question_system_prompt()
            prompts = [self.prompts.get_question_prompt(new_id, relation, identifier) for new_id, relation, identifier in zip(new_identifiers, relations, identifiers)]

        return system_prompts, prompts
    
    def _clean_json_block(self, item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def run(
            self,
            storage: DataFlowStorage,
            input_key:str = "generated_task",
            output_key:str = "generated_depth_task"
            ):
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        # step0: get identifier

        # Backward Step:
        # step1: Generate relation and superset
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "get_backward")
        backward_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
        
        identifiers = []
        relations = []
        valid_indices = []

        for idx, result in enumerate(backward_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))

                if isinstance(result, dict) and "identifier" in result and "relation" in result:
                    identifiers.append(result["identifier"])
                    relations.append(result["relation"])
                    valid_indices.append(idx)
                else:
                    self.logger.warning(f"[Skipped]: Result at index {idx} is invalid: {result}")
            except Exception as e:
                self.logger.warning(f"[Error]: Failed to parse backward result at index {idx}: {e}")
                continue

        dataframe = dataframe.iloc[valid_indices].copy()
        dataframe["new_identifier"] = identifiers
        dataframe["relation"] = relations

        # step2: Check if superset is valid
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "check_superset")
        check_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
        
        new_queries = []
        valid_indices = []
        for idx, result in enumerate(check_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))

                if isinstance(result, dict) and "new_query" in result:
                    new_queries.append(result["new_query"])
                    valid_indices.append(idx)
                else:
                    self.logger.warning(f"[Skipped]: Result at index {idx} is invalid: {result}")
            except Exception as e:
                self.logger.warning(f"[Error]: Failed to check superset result at index {idx}: {e}")
                continue

        dataframe = dataframe.iloc[valid_indices].copy()
        dataframe["new_query"] = new_queries

        # step3: Generate question based on superset and relation
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "get_new_question")
        check_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

        new_queries = []
        valid_indices = []
        for idx, result in enumerate(check_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))

                if isinstance(result, dict):
                    new_queries.append(result["new_query"])
                    valid_indices.append(idx)
                else:
                    self.logger.warning(f"[Skipped]: Result at index {idx} is invalid: {result}")
            except Exception as e:
                self.logger.warning(f"[Error]: Failed to check superset result at index {idx}: {e}")
                continue

        dataframe = dataframe.iloc[valid_indices].copy()
        dataframe["new_query"] = new_queries
        
        # Verify Step:
        # TODO: Search and Verify module

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return