from typing import List, Optional, Tuple, Union
import torch
def test(past_key_value: Optional[Tuple[torch.Tensor]] = None):
    print(type(past_key_value))
    print(past_key_value)
past_key_values=None
test(past_key_value=past_key_values)
