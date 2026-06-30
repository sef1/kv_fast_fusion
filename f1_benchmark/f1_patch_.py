from vllm.benchmarks.datasets import get_samples, add_dataset_parser  
  
def monkey_patch_dataset_mapping():  
    """Monkey patch to add custom_conversation dataset."""  
      
    # First, patch the argument parser to add our choice  
    original_add_dataset_parser = add_dataset_parser  
      
    def patched_add_dataset_parser(parser):  
        # Call original to add all standard arguments  
        original_add_dataset_parser(parser)  
          
        # Find the dataset-name argument and add our choice  
        for action in parser._actions:  
            if action.dest == 'dataset_name':  
                if 'custom_conversation' not in action.choices:  
                    action.choices.append('custom_conversation')  
                break  
      
    # Patch the argument parser function BEFORE it's used  
    import vllm.benchmarks.datasets  
    vllm.benchmarks.datasets.add_dataset_parser = patched_add_dataset_parser  
      
    # Now patch get_samples to handle our custom dataset  
    original_get_samples = get_samples  
      
    def patched_get_samples(args, tokenizer):  
        if args.dataset_name == "custom_conversation":  
            from f1_dataset import CustomConversationDataset  
              
            dataset = CustomConversationDataset(  
                dataset_path=args.dataset_path,  
                dataset_subset=getattr(args, 'hf_subset', None),  
                dataset_split=getattr(args, 'hf_split', 'train'),  
                input_key=getattr(args, 'input_key', 'problem'),  
                output_key=getattr(args, 'output_key', 'generated_solution'),  
                random_seed=args.seed,  
                disable_shuffle=args.disable_shuffle,  
            )  
              
            return dataset.sample(  
                tokenizer=tokenizer,  
                num_requests=args.num_prompts,  
                output_len=getattr(args, 'output_len', None),  
                request_id_prefix=getattr(args, 'request_id_prefix', ''),  
                no_oversample=getattr(args, 'no_oversample', False),  
            )  
          
        return original_get_samples(args, tokenizer)  
      
    vllm.benchmarks.datasets.get_samples = patched_get_samples  
  
# Apply the patch IMMEDIATELY when the module is imported  
monkey_patch_dataset_mapping()