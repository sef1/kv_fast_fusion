# vllm/benchmarks/agentic/benchmark_agentic.py  
import asyncio  
import json  
import time  
from dataclasses import dataclass  
from typing import Any, Dict, List  
  
import aiohttp  
from tqdm import tqdm  
  
from benchmarks.agentic.tool_call_dataset import ToolCallDataset  
from vllm.benchmarks.endpoint_request_func import (async_request_openai_chat_completions,
                                                   RequestFuncInput,
                                                   RequestFuncOutput)
# from benchmarks.lib.endpoint_request_func import (  
#     RequestFuncInput,  
#     RequestFuncOutput,  
#     async_request_openai_chat_completion,  
# )  
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ExtractedToolCallInformation
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.v1.core.sched import output  
from vllm.transformers_utils.tokenizer import get_tokenizer
from openai import OpenAI 
  
@dataclass  
class AgenticMetrics:  
    """Metrics for agentic AI benchmark."""  
    total_requests: int  
    successful_requests: int  
    tool_call_accuracy: float  
    avg_ttft_ms: float  
    avg_tpot_ms: float  
    tool_calls_per_second: float  
    token_efficiency: float  
  
  
class AgenticBenchmark:  
    """Benchmark for agentic AI capabilities with tool calling."""  
      
    def __init__(self, args):  
        self.args = args  
        self.dataset = ToolCallDataset(args.dataset)  
        # self.tool_parser = ToolParserManager.get_tool_parser(args.tool_parser)  
        tokenizer = get_tokenizer(args.model)
        tool_parser_class = ToolParserManager.get_tool_parser(args.tool_parser)  
        self.tool_parser = tool_parser_class(tokenizer)
        self.metrics = []  
          
    async def run_benchmark(self) -> AgenticMetrics:  
        """Run the agentic AI benchmark."""  
        print(f"Starting agentic AI benchmark with {self.args.num_prompts} prompts")  
          
        # Load test dataset  
        test_cases = self.dataset.get_test_cases(self.args.num_prompts)  
          
        # Initialize metrics collection  
        start_time = time.time()  
        results = []  
          
        # Run benchmark with progress bar  
        with tqdm(total=len(test_cases), desc="Running agentic benchmark") as pbar:  
            async with aiohttp.ClientSession() as session:  
                for test_case in test_cases:  
                    result = await self._execute_test_case(session, test_case)  
                    results.append(result)  
                    pbar.update(1)  
          
        # Calculate final metrics  
        total_time = time.time() - start_time  
        return self._calculate_metrics(results, total_time)  
      
    async def _execute_test_case(self, session, test_case) -> Dict[str, Any]:  
        """Execute a single test case."""  
        # prompt = test_case["messages"][-1]["content"] if test_case["messages"] else "" 
        # tokenizer = get_tokenizer(self.args.model)
        user_messages = [msg for msg in test_case["messages"] if msg["role"] == "user"]  
        
        request_input = RequestFuncInput(  
            model=self.args.model,  
            api_url=f"http://{self.args.host}:{self.args.port}/v1/chat/completions",  
            prompt="",  # Empty prompt since we'll override messages  
            prompt_len=0,  
            output_len=None,  
            extra_body={  
                "messages": user_messages, #test_case["messages"],  # Full message array with system/user roles  
                "tools": test_case.get("tools", []),  
                "tool_choice": "auto" if test_case.get("tools") else None,
                "stream": False,
                "temperature": None, 
                "stream_options": None,
            }  
        )
          
        start_time = time.time()  
        output: RequestFuncOutput = await async_request_openai_chat_completions(  
            request_input  
        )  
        end_time = time.time()  
          
        # Analyze tool call results  
        tool_call_result = self._analyze_tool_calls(  
            test_case, output, self.tool_parser  
        )  
          
        return {  
            "success": output.success,  
            "ttft": output.ttft,  
            "tpot": output.tpot if hasattr(output, 'tpot') else 0,  
            "latency": end_time - start_time,  
            "tool_call_correct": tool_call_result["correct"],  
            "expected_tools": len(test_case.get("tools", [])),  
            "actual_tools": len(tool_call_result["tool_calls"]),  
            "tokens_used": output.output_tokens,  
        }  
      
    def _analyze_tool_calls(self, test_case, output, tool_parser) -> Dict[str, Any]:  
        """Analyze if tool calls were executed correctly."""  
        if not test_case.get("tools"):  
            return {"correct": True, "tool_calls": []}  
        

         # Create a proper ChatCompletionRequest object  
        request = ChatCompletionRequest(  
        messages=test_case.get("messages", []),  
        model=self.args.model  
        )  
      
        # Extract tool calls from output  
        try:  
            if hasattr(output, 'generated_text'):  
                # CRITICAL: Always pass TWO arguments on a single line  
                tool_calls = tool_parser.extract_tool_calls(output.generated_text, request)  
            else:  
                tool_calls = []  
            
            # If pythonic parser didn't extract tool calls, try manual JSON parsing  
            if not tool_calls.tools_called and tool_calls.content:  
                try:  
                    # Parse the JSON string from content  
                    parsed_json = json.loads(tool_calls.content)  
                    if isinstance(parsed_json, dict) and "name" in parsed_json:  
                        # Create ToolCall object manually  
                        from vllm.entrypoints.openai.protocol import ToolCall, FunctionCall  
                        manual_tool_call = ToolCall(  
                            type="function",  
                            function=FunctionCall(  
                                name=parsed_json["name"],  
                                arguments=json.dumps(  
                                    parsed_json.get("parameters", parsed_json.get("arguments", {})),  
                                    ensure_ascii=False  
                                ),  
                            ),  
                        )  
                        tool_calls = ExtractedToolCallInformation(  
                            tools_called=True,  
                            tool_calls=[manual_tool_call],  
                            content=None  
                        )  
                except json.JSONDecodeError:  
                    # If it's not valid JSON, keep the original result  
                    pass  
            
            # Check if tool calls are correct  
            if tool_calls.tools_called:  
                expected_tools = {tool["function"]["name"] for tool in test_case["tools"]}  
                actual_tools = {call.function.name for call in tool_calls.tool_calls}  
                correct = expected_tools == actual_tools and tool_calls.tools_called  
            else:  
                correct = False  
            
            return {  
                "correct": correct,  
                "tool_calls": tool_calls.tool_calls,  
            }  
        except Exception as e:  
            print(f"Error analyzing tool calls: {e}")  
            return {"correct": False, "tool_calls": []}
        
    # def _analyze_tool_calls(self, test_case, output, tool_parser) -> Dict[str, Any]:  
    #     """Analyze if tool calls were executed correctly."""  
    #     if not test_case.get("tools"):  
    #         return {"correct": True, "tool_calls": []}  
        
    #     # Extract tool calls from output  
    #     try:  
    #         if hasattr(output, 'generated_text'):  
    #             # Must pass TWO arguments - model_output and request  
    #             tool_calls = tool_parser.extract_tool_calls(  
    #                 output.generated_text,  # First arg: model_output  
    #                 None                    # Second arg: request (can be None)  
    #             )  
    #         else:  
    #             tool_calls = []  
            
    #         # Check if expected tools were called  
    #         expected_tools = {tool["function"]["name"] for tool in test_case["tools"]}  
    #         actual_tools = {call.function.name for call in tool_calls.tool_calls}  
            
    #         correct = expected_tools == actual_tools and tool_calls.tools_called  
            
    #         return {  
    #             "correct": correct,  
    #             "tool_calls": tool_calls.tool_calls,  
    #         }  
    #     except Exception as e:  
    #         print(f"Error analyzing tool calls: {e}")  
    #         return {"correct": False, "tool_calls": []}
    
    def _calculate_metrics(self, results: List[Dict], total_time: float) -> AgenticMetrics:  
        """Calculate benchmark metrics from results."""  
        successful = [r for r in results if r["success"]]  
          
        if not successful:  
            return AgenticMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)  
          
        # Tool call accuracy  
        tool_correct = sum(1 for r in successful if r["tool_call_correct"])  
        tool_accuracy = tool_correct / len(successful) if successful else 0.0  
          
        # Latency metrics  
        avg_ttft = sum(r["ttft"] for r in successful) / len(successful)  
        avg_tpot = sum(r["tpot"] for r in successful) / len(successful)  
          
        # Throughput metrics  
        total_tool_calls = sum(r["actual_tools"] for r in successful)  
        tool_calls_per_sec = total_tool_calls / total_time if total_time > 0 else 0.0  
          
        # Token efficiency (tokens per tool call)  
        total_tokens = sum(r["tokens_used"] for r in successful)  
        token_efficiency = total_tokens / total_tool_calls if total_tool_calls > 0 else 0.0  
          
        return AgenticMetrics(  
            total_requests=len(results),  
            successful_requests=len(successful),  
            tool_call_accuracy=tool_accuracy,  
            avg_ttft_ms=avg_ttft * 1000,  
            avg_tpot_ms=avg_tpot * 1000,  
            tool_calls_per_second=tool_calls_per_sec,  
            token_efficiency=token_efficiency,  
        )  
  
  
def main(args):  
    """Main entry point for agentic benchmark."""  
    benchmark = AgenticBenchmark(args)  
      
    async def run():  
        metrics = await benchmark.run_benchmark()  
          
        print("\n" + "="*60)  
        print("Agentic AI Benchmark Results")  
        print("="*60)  
        print(f"Total requests: {metrics.total_requests}")  
        print(f"Successful requests: {metrics.successful_requests}")  
        print(f"Tool call accuracy: {metrics.tool_call_accuracy:.2%}")  
        print(f"Average TTFT: {metrics.avg_ttft_ms:.2f} ms")  
        print(f"Average TPOT: {metrics.avg_tpot_ms:.2f} ms")  
        print(f"Tool calls/second: {metrics.tool_calls_per_second:.2f}")  
        print(f"Token efficiency: {metrics.token_efficiency:.2f} tokens/tool")  
        print("="*60)  
      
    asyncio.run(run())