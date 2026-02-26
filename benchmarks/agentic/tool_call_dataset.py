# vllm/benchmarks/agentic/tool_call_dataset.py  
import json  
import random  
from typing import Dict, List, Any  
  
  
class ToolCallDataset:  
    """Dataset for agentic AI benchmark with tool calling scenarios."""  
      
    def __init__(self, dataset_name: str = "tool_calling"):  
        self.dataset_name = dataset_name  
        self._load_tools()  
      
    def _load_tools(self):  
        """Load tool definitions for benchmarking."""  
        self.tools = {  
            "get_weather": {  
                "type": "function",  
                "function": {  
                    "name": "get_weather",  
                    "description": "Get the current weather in a location",  
                    "parameters": {  
                        "type": "object",  
                        "properties": {  
                            "location": {  
                                "type": "string",  
                                "description": "The city and state, e.g. San Francisco, CA"  
                            },  
                            "unit": {  
                                "type": "string",  
                                "enum": ["celsius", "fahrenheit"],  
                                "description": "Temperature unit"  
                            }  
                        },  
                        "required": ["location"]  
                    }  
                }  
            },  
            "search_web": {  
                "type": "function",  
                "function": {  
                    "name": "search_web",  
                    "description": "Search the web for information",  
                    "parameters": {  
                        "type": "object",  
                        "properties": {  
                            "query": {  
                                "type": "string",  
                                "description": "Search query"  
                            },  
                            "num_results": {  
                                "type": "integer",  
                                "description": "Number of results to return",  
                                "default": 5  
                            }  
                        },  
                        "required": ["query"]  
                    }  
                }  
            },  
            "calculate": {  
                "type": "function",  
                "function": {  
                    "name": "calculate",  
                    "description": "Perform mathematical calculations",  
                    "parameters": {  
                        "type": "object",  
                        "properties": {  
                            "expression": {  
                                "type": "string",  
                                "description": "Mathematical expression to evaluate"  
                            }  
                        },  
                        "required": ["expression"]  
                    }  
                }  
            }  
        }  
      
    def get_test_cases(self, num_cases: int) -> List[Dict[str, Any]]:  
        """Generate test cases for the benchmark."""  
        test_cases = []  
          
        # Simple tool call scenarios
        simple_cases = [    
            {
            "messages": [
                # {"role": "system", "content": "You are an intelligent assistant ready to help."},
                {"role": "user", "content": "What's the weather like in London?"},
                {"role": "assistant", "content": "I'll check the weather for you.", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"London\", \"unit\": \"celsius\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\": \"18°C\", \"condition\": \"partly cloudy\"}"},
                {"role": "assistant", "content": "The weather in London is 18°C and partly cloudy."},
                {"role": "user", "content": "Search for attractions there."},
                {"role": "assistant", "content": "Let me find some attractions for you.", "tool_calls": [
                    {"id": "call_2", "type": "function", "function": {"name": "search_web", "arguments": "{\"query\": \"London attractions\", \"num_results\": \"5\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_2", "content": "{\"results\": [\"The British Museum\", \"The Tower of London\", \"The London Eye\", \"Buckingham Palace\", \"The National Gallery\"]}"},
                {"role": "assistant", "content": "Here are some attractions in London: The British Museum, The Tower of London, The London Eye, Buckingham Palace, and The National Gallery."},
                {"role": "user", "content": "What's the history behind The Tower of London?"},
                {"role": "assistant", "content": "I can look that up for you.", "tool_calls": [
                    {"id": "call_3", "type": "function", "function": {"name": "search_web", "arguments": "{\"query\": \"History of the Tower of London\", \"num_results\": \"1\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_3", "content": "{\"summary\": \"The Tower of London has a rich history, serving as a royal palace, prison, and treasury since its construction in 1066.\"}"},
                {"role": "assistant", "content": "The Tower of London has a rich history, serving as a royal palace, prison, and treasury since its construction in 1066."},
                {"role": "user", "content": "can you tell how many days passed since the Tower of London construction ?"},
                {"role": "assistant", "content": "Let me calculate that for you.", "tool_calls": [
                        {
                            "id": "call_124",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": "{\"expression\": \"(365 * (2025 - 1066))\"}"
                            }
                        }
                    ]},
                {"role": "tool", "tool_call_id": "call_124", "content": "{\"result\": \"315365\"}"},
                {"role": "assistant", "content": "The Tower of London has been standing for 315,365 days."}
            ],
            "tools": [
                self.tools["get_weather"],
                self.tools["search_web"],
                self.tools["calculate"]
            ],
            "expected_tools": ["get_weather", "search_web", "calculate"]
           }
        ]
      
        # simple_cases = [  
        # {  
        #     "messages": [  
        #         {"role": "user", "content": "What's the weather like in New York?"},  
        #         {"role": "assistant", "content": "I'll check the weather for you.", "tool_calls": [  
        #             {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "New York", "unit": "fahrenheit"}'}}  
        #         ]},  
        #         {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature": "72°F", "condition": "sunny"}'},  
        #         {"role": "assistant", "content": "The weather in New York is 72°F and sunny."}  
        #     ],  
        #     "tools": [self.tools["get_weather"]],  
        #     "expected_tools": ["get_weather"]  
        # }, 
        # {  
        #     "messages": [  
        #         {"role": "user", "content": "What's the weather like in New York?"},  
        #         {"role": "assistant", "content": "I'll check the weather for you.", "tool_calls": [  
        #             {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "New York", "unit": "fahrenheit"}'}}  
        #         ]},  
        #         {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature": "60°F", "condition": "sunny"}'},  
        #         {"role": "assistant", "content": "The weather in New York is 60°F and sunny."}  
        #     ],  
        #     "tools": [self.tools["get_weather"]],  
        #     "expected_tools": ["get_weather"]  
        # },   
        # ]  
        
        # Multi-tool scenarios  
        # multi_tool_cases = [  
        #     {  
        #         "messages": [  
        #             {"role": "system", "content": "You are a  helpful assistant with access to tools. When asked about weather, search, or calculations, use the appropriate tool."},  
        #             {"role": "user", "content": "What's the weather in London and then search for tourist attractions there?"}  
        #         ],  
        #         "tools": [self.tools["get_weather"], self.tools["search_web"]],  
        #         "expected_tools": ["get_weather", "search_web"]  
        #     }  
        # ]  

        # multi_tool_cases = [  
        #     {  
        #         "messages": [  
        #             # {"role": "system", "content": "You are a helpful assistant with access to tools. When asked about weather, search, or calculations, use the appropriate tool."},  
        #             {"role": "user", "content": "What's the weather in London and then search for tourist attractions there?"},  
        #             {  
        #                 "role": "assistant",  
        #                 "tool_calls": [  
        #                     {  
        #                         "id": "call_1",  
        #                         "type": "function",  
        #                         "function": {  
        #                             "name": "get_weather",  
        #                             "arguments": '{"location": "London", "unit": "celsius"}'  
        #                         }  
        #                     },  
        #                     {  
        #                         "id": "call_2",   
        #                         "type": "function",  
        #                         "function": {  
        #                             "name": "search_web",  
        #                             "arguments": '{"query": "London tourist attractions", "num_results": "10"}'  
        #                         }  
        #                     }  
        #                 ]  
        #             },  
        #             {"role": "tool", "tool_call_id": "call_1", "content": "The weather in London is 15°C, partly cloudy."},  
        #             {"role": "tool", "tool_call_id": "call_2", "content": "Top London attractions: Big Ben, Tower Bridge, Buckingham Palace, London Eye, British Museum..."},  
        #             {"role": "assistant", "content": "Based on the weather and attractions, London is currently 15°C and partly cloudy. Great places to visit include Big Ben, Tower Bridge, Buckingham Palace, the London Eye, and the British Museum."}  
        #         ],  
        #         "tools": [self.tools["get_weather"], self.tools["search_web"]],  
        #         "expected_tools": ["get_weather", "search_web"]  
        #     }  
        # ]  
        multi_tool_cases = []
          
        # Multi-turn conversations  
        # multi_turn_cases = [  
        #     {  
        #         "messages": [  
        #             # {"role": "system", "content": "You are a helpful assistant with access to tools. When asked about weather, search, or calculations, use the appropriate tool."},  
        #             {"role": "user", "content": "What's the weather in Tokyo?"},  
        #             {"role": "assistant", "content": "I'll check the weather in Tokyo for you.", "tool_calls": [  
        #                 {  
        #                     "id": "call_123",  
        #                     "type": "function",  
        #                     "function": {  
        #                         "name": "get_weather",   
        #                         "arguments": '{"location": "Tokyo", "unit": "celsius"}'  
        #                     }  
        #                 }  
        #             ]},  
        #             {"role": "tool", "tool_call_id": "call_123", "content": '{"temperature": "22°C", "condition": "sunny"}'},  
        #             {"role": "assistant", "content": "The weather in Tokyo is 22°C and sunny."},  
        #             {"role": "user", "content": "That's nice! Can you convert that to Fahrenheit?"}  
        #         ],  
        #         "tools": [self.tools["calculate"]],  
        #         "expected_tools": ["calculate"]  
        #     }  
        # ]
        multi_turn_cases = [
            {
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant with access to tools. When asked about weather, search, or calculations, use the appropriate tool."},
                    {"role": "user", "content": "What's the weather in Tokyo?"},
                    {"role": "assistant", "content": "I'll check the weather in Tokyo for you.", "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
                            }
                        }
                    ]},
                    {"role": "tool", "tool_call_id": "call_123", "content": "{\"temperature\": \"22°C\", \"condition\": \"sunny\"}"},
                    {"role": "assistant", "content": "The weather in Tokyo is 22°C and sunny."},
                    {"role": "user", "content": "That's nice! Can you convert that to Fahrenheit?"}, 
                    {"role": "assistant", "content": "Let me calculate that for you.", "tool_calls": [
                        {
                            "id": "call_124",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": "{\"expression\": \"(22 * 9/5) + 32\"}"
                            }
                        }
                    ]},
                    {"role": "tool", "tool_call_id": "call_124", "content": "{\"result\": \"71.6\"}"},
                    {"role": "assistant", "content": "The temperature in Fahrenheit is approximately 71.6°F."},
                    {"role": "user", "content": "What should I wear today?"},  
                    {"role": "assistant", "content": "Since it's sunny at 22°C, you might want to wear light clothing and sunglasses."},  
                    {"role": "user", "content": "Are there any events happening in Tokyo today?"},  
                    {"role": "assistant", "content": "I'll check for events happening in Tokyo.", "tool_calls": [
                        {
                            "id": "call_125",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": "{\"query\": \"events today in Tokyo\", \"num_results\": \"5\"}"
                            }
                        }
                    ]},
                    {"role": "tool", "tool_call_id": "call_125", "content": "{\"results\": [\"Tokyo Hanabi Festival\", \"Tokyo Art Museum Exhibition\", \"Cherry Blossom Viewing at Ueno Park\"]}"},
                    {"role": "assistant", "content": "Here are some events happening in Tokyo today: Tokyo Hanabi Festival, Tokyo Art Museum Exhibition, and Cherry Blossom Viewing at Ueno Park."}
                ],
                "tools": [
                    self.tools["get_weather"],
                    self.tools["calculate"],
                    self.tools["search_web"]
                ],
                "expected_tools": ["get_weather", "search_web", "book_ticket"]
            }

        ]
                
        # Error handling cases  
        # error_cases = [  
        #     {  
        #         "messages": [  
        #             {"role": "system", "content": "You are a helpful assistant with access to tools. When asked about weather, search, or calculations, use the appropriate tool."},  
        #             {"role": "user", "content": "What's the weather?"}  
        #         ],  
        #         "tools": [self.tools["get_weather"]],  
        #         "expected_tools": []  # Should fail due to missing location  
        #     }  
        # ]  
        error_cases = [  
            # {  
            #     "messages": [  
            #         # {"role": "system", "content": "You are a helpful assistant with access to tools. When asked about weather, search, or calculations, use the appropriate tool."},  
            #         {"role": "user", "content": "What's the weather?"},  
            #         {  
            #             "role": "assistant",  
            #             "content": "I'd be happy to help you get the weather information. However, I need to know which location you're asking about. Could you please specify the city or location for which you'd like the weather?"  
            #         }  
            #     ],  
            #     "tools": [self.tools["get_weather"]],  
            #     "expected_tools": []  # Should fail due to missing location  
            # }  
            {
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant with access to tools. When asked about weather, search, or calculations, use the appropriate tool."},
                    {"role": "user", "content": "What's the weather?"},
                    {
                        "role": "assistant",
                        "content": "I'd be happy to help you get the weather information. However, I need to know which location you're asking about. Could you please specify the city or location for which you'd like the weather?"
                    },
                    {"role": "user", "content": "I meant New York."},
                    {"role": "assistant", "content": "Thank you for clarifying! I'll check the weather for New York.", "tool_calls": [
                        {
                            "id": "call_126",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"New York\", \"unit\": \"fahrenheit\"}"
                            }
                        }
                    ]},
                    {"role": "tool", "tool_call_id": "call_126", "content": "{\"temperature\": \"72°F\", \"condition\": \"sunny\"}"},
                    {"role": "assistant", "content": "The weather in New York is 72°F and sunny."},
                    {"role": "user", "content": "What about Los Angeles?"},  
                    {"role": "assistant", "content": "I'll check the weather for Los Angeles.", "tool_calls": [
                        {
                            "id": "call_127",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"Los Angeles\", \"unit\": \"fahrenheit\"}"
                            }
                        }
                    ]},
                    {"role": "tool", "tool_call_id": "call_127", "content": "{\"temperature\": \"75°F\", \"condition\": \"clear\"}"},
                    {"role": "assistant", "content": "The weather in Los Angeles is 75°F and clear."},
                    {"role": "user", "content": "Can you find out the weather in Paris?"},  
                    {"role": "assistant", "content": "I'll check the weather for Paris.", "tool_calls": [
                        {
                            "id": "call_128",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"
                            }
                        }
                    ]},
                    {"role": "tool", "tool_call_id": "call_128", "content": "{\"temperature\": \"21°C\", \"condition\": \"overcast\"}"},
                    {"role": "assistant", "content": "The weather in Paris is 21°C and overcast."}
                ],
                "tools": [self.tools["get_weather"]],
                "expected_tools": []
            }
    ]
          
        # Combine all cases  
        all_cases = simple_cases + multi_tool_cases + multi_turn_cases + error_cases  
          
        # Generate requested number of cases by cycling through available cases  
        for i in range(num_cases):  
            base_case = all_cases[i % len(all_cases)]  
            # Create a copy to avoid modifying the original  
            test_case = json.loads(json.dumps(base_case))  
            test_cases.append(test_case)  
          
        return test_cases
        