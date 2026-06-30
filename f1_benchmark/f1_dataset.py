from vllm.benchmarks.datasets import HuggingFaceDataset, SampleRequest  
from vllm.tokenizers import TokenizerLike  
from typing import List  
  
class CustomConversationDataset(HuggingFaceDataset):  
    """Dataset for loading conversation data from HuggingFace datasets.  
    Loads prompt-completion pairs and generates sample requests.  
    """  
      
    SUPPORTED_DATASET_PATHS = {  
        "nvidia/OpenMathInstruct-2",  
    }  
      
    def __init__(  
        self,  
        input_key: str = "problem",  
        output_key: str = "generated_solution",  
        min_input_len: int = 257,  
        max_input_len: int = 8192,  
        max_total_len: int = 128 * 1024,  
        **kwargs,  
    ) -> None:  
        self.input_key = input_key  
        self.output_key = output_key  
        self.min_input_len = min_input_len  
        self.max_input_len = max_input_len  
        self.max_total_len = max_total_len 
        self.ground_truth_map = {} 
        super().__init__(**kwargs)  
      
    def sample(  
        self,  
        tokenizer: TokenizerLike,  
        num_requests: int,  
        output_len: int | None = None,  
        enable_multimodal_chat: bool = False,  
        request_id_prefix: str = "",  
        no_oversample: bool = False,  
        **kwargs,  
    ) -> List[SampleRequest]:  
        sampled_requests = []  
          
        for i, item in enumerate(self.data):  
            if len(sampled_requests) >= num_requests:  
                break  
              
            # Extract prompt and completion  
            prompt = item[self.input_key]  
            completion = item[self.output_key]  
              
            if completion is None:  
                continue  
            if isinstance(completion, list):  
                completion = completion[0]  
              
            # Tokenize  
            prompt_token_ids = tokenizer(prompt).input_ids  
            completion_token_ids = tokenizer(completion).input_ids  
            prompt_len = len(prompt_token_ids)  
            output_len_actual = len(completion_token_ids)  
              
            # Filter by length constraints  
            if prompt_len < self.min_input_len:  
                continue  
            if prompt_len > self.max_input_len:  
                continue  
            if prompt_len + output_len_actual > self.max_total_len:  
                continue  
              
            # Apply chat template  
            prompt_formatted = tokenizer.apply_chat_template(  
                [{"role": "user", "content": prompt}],  
                add_generation_prompt=True,  
                tokenize=False,  
            )  

            request_id = request_id_prefix + str(i)  
            self.ground_truth_map[request_id] = completion
              
            sampled_requests.append(  
                SampleRequest(  
                    prompt=prompt_formatted,  
                    prompt_len=prompt_len,  
                    expected_output_len=output_len_actual,  
                    request_id=request_id_prefix + str(i),  
                )  
            )  
          
        self.maybe_oversample_requests(  
            sampled_requests, num_requests, request_id_prefix, no_oversample  
        )  
        return sampled_requests