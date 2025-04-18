import torch 
import torch.nn.functional as F

class Bi_gram():
    def __init__(self , words):
        self.chars = set(list(''.join(words)))
        self.n = len(self.chars) + 1
        words = ['$'+word+'$' for word in words]
        self.data = []
        for word in words:
            for ch1 , ch2 in zip(word, word[1:]):
                self.data.append(ch1+ch2)

        self.char2idx = {char:idx+1 for idx, char in enumerate(self.chars)}
        self.idx2char = {idx+1:char for idx, char in enumerate(self.chars)}
        self.char2idx['$'] = 0
        self.idx2char[0] = '$'


        self.w = torch.rand((self.n, self.n), requires_grad=True)
  

    def train(self, num_iterations = 1000):

      x = torch.tensor([self.char2idx[ch[0]] for ch in self.data])
      x = F.one_hot(x, num_classes=self.n).float()
      y = torch.tensor([self.char2idx[ch[-1]] for ch in self.data])

      for i in range(num_iterations):
        logits = x @ self.w
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(len(probs)), y].log().mean()

        self.w.grad = None
        loss.backward()
        self.w.data += -0.1 * self.w.grad
    
        

    def predict(self, num_predictions=10):
        results = []
        for i in range(num_predictions):
            start = '$' # initial char
            result = ''
            while True:
                x = torch.tensor(self.char2idx[start])
                x = F.one_hot(x, num_classes=self.n).float().reshape(-1, self.n) # one hot encoding
                logits = x @ self.w  # forward pass
                probs = torch.softmax(logits, dim=1)
                next_idx = torch.multinomial(probs, num_samples=1).item() # sampling from the result distribution
                next_char = self.idx2char[next_idx]
                
                if next_char == '$':
                    break
                result += next_char
                start = next_char

            results.append(result)
        return results