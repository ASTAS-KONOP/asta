from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TRAIN_TEXT_FILE_PATH = 'm.txt'

with open(TRAIN_TEXT_FILE_PATH) as text_file:
    text_sample = text_file.readlines()
text_sample = ' '.join(text_sample)

def text_to_seq(text_sample):
    bukv = Counter(text_sample)
    bukv = sorted(bukv.items(), key = lambda x: x[1], reverse=True)

    s_bukva = [char for char, _ in bukv]
    print(s_bukva)
    b_to_d = {char: index for index, char in enumerate(s_bukva)}
    d_to_b = {v: k for k, v in b_to_d.items()}
    sequence = np.array([b_to_d[char] for char in text_sample])
    
    return sequence, b_to_d, d_to_b

sequence, b_to_d, d_to_b = text_to_seq(text_sample)

SEQ_LEN = 256
BATCH_SIZE = 160

def get_batch(sequence):
    trains = []
    targets = []
    for _ in range(BATCH_SIZE):
        batch_start = np.random.randint(0, len(sequence) - SEQ_LEN)
        chunk = sequence[batch_start: batch_start + SEQ_LEN]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)


def evaluate(model, b_to_d, d_to_b, start_text=' ', prediction_len=200, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [b_to_d[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text
    
    _, hidden = model(train, hidden)
        
    inp = train[-1].view(-1, 1, 1)
    
    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()        
        top_index = np.random.choice(len(b_to_d), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        p_bukva = d_to_b[top_index]
        predicted_text += p_bukva
    
    return predicted_text

class TextRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        
    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)
    
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
               torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = TextRNN(input_size=len(d_to_b), hidden_size=1280, embedding_size=128, n_layers=5)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    patience=5, 
    verbose=True, 
    factor=0.5
)

n_epochs = 50000
loss_avg = []

for epoch in range(n_epochs):
    model.train()
    train, target = get_batch(sequence)
    train = train.permute(1, 0, 2).to(device)
    target = target.permute(1, 0, 2).to(device)
    hidden = model.init_hidden(BATCH_SIZE)

    output, hidden = model(train, hidden)
    loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    loss_avg.append(loss.item())
    if len(loss_avg) >= 50:
        mean_loss = np.mean(loss_avg)
        print(f'Loss: {mean_loss}')
        scheduler.step(mean_loss)
        loss_avg = []
        model.eval()
        predicted_text = evaluate(model, b_to_d, d_to_b)
        print(predicted_text)

torch.save(model.state_dict(), 'setochka.py')

model.eval()
print(evaluate(
    model, 
    b_to_d, 
    d_to_b, 
    temp=0.9, 
    prediction_len=900, 
    start_text=' '
    )
)





