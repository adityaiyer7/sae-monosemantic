import torch
from transformers import GPT2Tokenizer, GPT2Model

class GPT2Wrapper:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained('gpt2').eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def encode(self, input_lst):
        encoded_input = self.tokenizer(input_lst, return_tensors='pt', padding= 'max_length', truncation=True, max_length=1024)
        return encoded_input.to(self.device)

    @torch.no_grad()
    def forward(self, encoded_input, layer = 6):
        output = self.model(**encoded_input, output_hidden_states=True)
        return output.hidden_states[layer]
    
# example
# text = "I am a rock"
# gpt2_small = GPT2Wrapper()
# encoded = gpt2_small.encode(text)
# residual_state = gpt2_small.forward(encoded, 5) 
# print(residual_state)
# print(type(residual_state))
# print(residual_state.size())




