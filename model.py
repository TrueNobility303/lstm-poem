import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn 
import tqdm
import matplotlib.pyplot as plt 
import numpy as np

# REF: https://github.com/chenyuntc/pytorch-book/blob/master/chapter09-neural_poet_RNN/model.py

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        seq_len, batch_size = inputs.size()
        if hidden is None:
            h0 = inputs.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c0 = inputs.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h0, c0 = hidden

        embeds = self.embeddings(inputs)
        output, hidden = self.lstm(embeds, (h0, c0))
        output = self.linear(output.view(seq_len * batch_size, -1))

        return output, hidden

class PoetryGererater():
    def __init__(self):
        self.word2ix = np.load("./datasets/tang5/word2ix.npy", allow_pickle=True).item()
        self.ix2word = np.load("./datasets/tang5/ix2word.npy", allow_pickle=True).item()

        #print(self.word2ix['平'])
        #print(self.word2ix['安'])
        #print(self.word2ix['復'])
        #print(self.word2ix['旦'])

        self.model = PoetryModel(len(self.word2ix), 128, 256)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.epoch = 20

    def train(self):
        data = np.load("./datasets/tang5/data.npy", allow_pickle=True)
        data = torch.from_numpy(data)
        data = data.to(device)
        dataloader = torch.utils.data.DataLoader(data,batch_size=128,shuffle=True)

        self.model.to(device)

        LossList = []
        for epoch in tqdm.tqdm(range(self.epoch)):
            L = 0
            for i, data in enumerate(dataloader):
                data = data.long().transpose(1, 0).contiguous()
                self.optimizer.zero_grad()
                inputs, target = data[:-1, :], data[1:, :]
                output, _ = self.model(inputs)
                loss = self.criterion(output, target.view(-1))
                L += loss.item()
                loss.backward()
                self.optimizer.step()
            LossList.append(L)
            if epoch % 1 == 0:
                print(self.generate())

        plt.plot(LossList)
        plt.savefig("./dump/loss.png")
        #plt.show()

    def generate(self,start_words="平安復旦"):
        
        results = ""
        
        for sw in start_words:
            hidden = None
            inputs = torch.LongTensor([self.word2ix['<START>']]).view(1, 1).to(device)
            output, hidden = self.model(inputs, hidden)

            sentence = ""
            w = str(sw)
            for i in range(12):
                sentence += w
                inputs = torch.LongTensor([self.word2ix[w]]).view(1, 1).to(device)
                output, hidden = self.model(inputs, hidden)
                top_index = output.data[0].topk(1)[1][0].item()
                w = self.ix2word[top_index]
            results += sentence
        return results

def main(training):
    PG = PoetryGererater()
    if training is True:
        PG.train()
        torch.save(PG.model.state_dict(), "./checkpoints/model5.pth")
    else:
        PG.model.load_state_dict(torch.load("./checkpoints/model5.pth"))
        print(PG.generate()) 

if __name__ == "__main__":
    main(training=True)
               

                