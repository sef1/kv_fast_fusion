from abc import ABC, abstractmethod  
from typing import Any  
import torch  
  
class CompressionHook(ABC):  
    @abstractmethod  
    def start_layer_compression(  
        self,   
        layer_name: str,  
        kv_cache: torch.Tensor,  
        attn_metadata: Any,  
        **kwargs: Any  
    ) -> None:  
        """Start async compression for a specific layer."""  
        pass  
      
    @abstractmethod  
    def wait_for_layer_compression(self, layer_name: str) -> None:  
        """Wait for compression to complete for a layer."""  
        pass