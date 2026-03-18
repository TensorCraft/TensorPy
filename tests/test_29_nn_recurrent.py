import ml
import nn


def ok(x, m):
    if not x:
        raise RuntimeError(m)


t = ml.tensor([-1, 0, 1]).tanh()
ok(t.max() > 0.7, "a")
ok(t.max() < 0.8, "b")

embedding = nn.Embedding(6, 4)
embedded = embedding.forward([[0, 1], [2, 3]])
ok(embedded.rank == 3, "c")
ok(embedded.shape[0] == 2 and embedded.shape[1] == 2 and embedded.shape[2] == 4, "d")

block = nn.Sequential([
    nn.Linear(2, 3),
    nn.ReLU(),
])
params = block.parameters()
ok(len(params) == 2, "e")

rnn = nn.RNN(1, 4)
rnn_in = ml.tensor([
    [[1], [2], [3]],
    [[2], [3], [4]],
])
outputs, h = rnn.forward(rnn_in)
ok(outputs.rank == 3, "f")
ok(outputs.shape[0] == 2 and outputs.shape[1] == 3 and outputs.shape[2] == 4, "g")
ok(h.rank == 3, "h")
ok(h.shape[0] == 1 and h.shape[1] == 2 and h.shape[2] == 4, "i")

lstm = nn.LSTM(1, 3)
lstm_outputs, lstm_state = lstm.forward(rnn_in)
ok(lstm_outputs.rank == 3, "j")
ok(lstm_outputs.shape[0] == 2 and lstm_outputs.shape[1] == 3 and lstm_outputs.shape[2] == 3, "k")
ok(lstm_state[0].shape[0] == 1 and lstm_state[0].shape[1] == 2 and lstm_state[0].shape[2] == 3, "l")
ok(lstm_state[1].shape[0] == 1 and lstm_state[1].shape[1] == 2 and lstm_state[1].shape[2] == 3, "m")

gru = nn.GRU(1, 5)
gru_outputs, gru_h = gru.forward(rnn_in)
ok(gru_outputs.rank == 3, "n")
ok(gru_outputs.shape[0] == 2 and gru_outputs.shape[1] == 3 and gru_outputs.shape[2] == 5, "o")
ok(gru_h.shape[0] == 1 and gru_h.shape[1] == 2 and gru_h.shape[2] == 5, "p")

print("nn-recurrent-ok")
