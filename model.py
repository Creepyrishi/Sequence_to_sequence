import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size_eng, embedding_size, hidden_size, num_layer, p):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size_eng
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layer, dropout=p)
    
    def forward(self, X):
        embed = self.embed(X)
        # Shape (vocab_size, embedding_size)
        output, (hn, cn)  = self.rnn(embed)
        # Shape of hn, cn (no of direction * layers of LSTM, batch size, no. of nodes in lstm)
        # Shape of output (no. of token (time steps), batch size, no. of nodes * no. of direction)
        return hn, cn
    

class Decoder(nn.Module):
    def __init__(self, vocab_size_french, embedding_size, hidden_size, num_layer, p):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size_french
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layer, dropout=p)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

    def forward(self, X, hidden, cell):
        # Decoder will work word by word
        # The dimention of the X will be N but we need 1, N
        if X.dim() == 1:
            X = X.unsqueeze(0)
        embed = self.embed(X)


        output, (hn, cn) = self.rnn(embed, (hidden, cell))

        prediction = self.fc(output)
        prediction = prediction.squeeze(0)

        return prediction, hn, cn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    
    def forward(self, source, target):

        # Shape of target (no of words including <sos> and <eos> or no of token, batch size)
        # Shape of source (no of words including <sos> and <eos> or no of token, batch size)

        hn, cn = self.encoder(source)

        no_of_token_in_target, batch_size = target.shape
        outputs = torch.zeros(size=(no_of_token_in_target,  batch_size, self.decoder.vocab_size), device=source.device)
        # Shape of outputs (no of token, batch size, vocab_size_frech)

        teach = target[0] # Setting first word
        for t in range(1, no_of_token_in_target):
                    
            # print("Input shape:", teach.shape)
            # print("Hidden shape:", hn.shape)
            # print("Cell shape:", cn.shape)
            prediction, hn, cn = self.decoder(teach, hn, cn)
            outputs[t] = prediction
            teach = target[t, :] # Gets word in 0 postion from all sentences or sequence from whole batch [get the ground truth for teacher forcing

        return outputs