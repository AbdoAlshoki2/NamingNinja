import torch 
import torch.nn.functional as F

class N_gram():
    def __init__(self, words, previous_context: int = 3 , embedding_size : int = 2):
        self.chars = set(list(''.join(words)))
        self.n = len(self.chars) + 1
        self.emb_size = embedding_size
        self.prv_cntxt = previous_context
        
        self.data = []
        for word in words:
            start = '$' * previous_context
            for ch in word + '$':
                self.data.append(start + ch)
                start = start[1:] + ch

        self.char2idx = {char:idx+1 for idx, char in enumerate(self.chars)}
        self.idx2char = {idx+1:char for idx, char in enumerate(self.chars)}
        self.char2idx['$'] = 0
        self.idx2char[0] = '$'


        self.C = torch.rand((self.n , self.emb_size), requires_grad=True)

    def train(self, num_iterations=100, batch_size=32):

        y_full = torch.tensor([self.char2idx[ch[-1]] for ch in self.data])
        x_indices_full = torch.tensor([[self.char2idx[c] for c in ch[:-1]] for ch in self.data])
        
        self.w1 = torch.rand((self.prv_cntxt * self.emb_size, 100), requires_grad=True)
        self.b1 = torch.rand(100, requires_grad=True)
        self.w2 = torch.rand((100, self.n), requires_grad=True)
        self.b2 = torch.rand(self.n, requires_grad=True)
        self.parameters = [self.C, self.w1, self.b1, self.w2, self.b2]
        
        num_samples = len(y_full)
        num_batches = (num_samples + batch_size - 1) // batch_size 
        
        for i in range(num_iterations):

            indices = torch.randperm(num_samples)
            x_indices_shuffled = x_indices_full[indices]
            y_shuffled = y_full[indices]
            
            total_loss = 0

            for b in range(num_batches):

                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, num_samples)
                
                x_indices_batch = x_indices_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                for p in self.parameters:
                    if p.grad is not None:
                        p.grad.zero_()
                

                x_batch = self.C[x_indices_batch].view(-1, self.prv_cntxt * self.emb_size)
                logits = x_batch @ self.w1 + self.b1
                h = torch.tanh(logits)
                logits = h @ self.w2 + self.b2
                counts = logits.exp()
                probs = counts / counts.sum(1, keepdims=True)
                loss = -probs[torch.arange(len(probs)), y_batch].log().mean()
                total_loss += loss.item()

                loss.backward()

                for p in self.parameters:
                    if p.grad is not None:
                        p.data -= 0.1 * p.grad

    
    def predict(self, num_predictions=10):
        results = []
        for i in range(num_predictions):
            start = '$' * self.prv_cntxt
            result = ''
            while True:
                x = torch.tensor([[self.char2idx[c] for c in start]])
                x = self.C[x].view(-1, self.prv_cntxt * self.emb_size)
                h = torch.tanh(x @ self.w1 + self.b1)
                logits = h @ self.w2 + self.b2
                probs = torch.softmax(logits, dim=1)
                next_idx = torch.multinomial(probs, num_samples=1).item() # sampling from the result distribution
                next_char = self.idx2char[next_idx]
                if next_char == '$':
                    break
                result += next_char
                start = start[1:] + next_char

            results.append(result)
        return results

with open('names.txt') as f:
  lst = f.read().splitlines()

n = N_gram(lst)
