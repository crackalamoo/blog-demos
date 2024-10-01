import numpy as np
import matplotlib.pyplot as plt
import torch
import time

STEP = 0.1
X_seq = np.arange(0, 5000, STEP)
seq = np.sin(X_seq) + 1/5*np.cos(11/13*X_seq) + 1/9*np.sin(17/37*X_seq-np.pi/4)
raw_seq = seq.copy()
seq += 1/7*np.random.normal(size=seq.size)

X = []
y = []
for i in range(25, len(X_seq)):
    X.append(seq[i-25:i])
    y.append(seq[i])
X = np.array(X)
y = np.array(y)
test_len = max(200, len(X)//10)
X_train = X[:-test_len]
y_train = y[:-test_len]
X_test = X[-test_len:]
y_test = y[-test_len:]

X_train = torch.tensor(X_train).unsqueeze(2)
y_train = torch.tensor(y_train).unsqueeze(1)
X_test = torch.tensor(X_test).unsqueeze(2)
y_test = torch.tensor(y_test).unsqueeze(1)
X_train = X_train.float()
y_train = y_train.float()
X_test = X_test.float()
y_test = y_test.float()

start_transformer = time.time()
embed = torch.nn.Linear(1, 4)
transformer = torch.nn.TransformerDecoder(
    torch.nn.TransformerDecoderLayer(4, 1, 8, 0, 'relu', batch_first=True),
    1
)
unembed = torch.nn.Linear(4, 1)
optim = torch.optim.Adam(transformer.parameters(), lr=0.01) # 0.1
criterion = torch.nn.MSELoss()

losses = []
for epoch in range(1):
    batches = torch.randperm(len(X_train))
    batch_size = 128
    for i in range(0, len(batches), batch_size):
        batch = batches[i:i+batch_size]
        optim.zero_grad()
        embeds = embed(X_train[batch])
        y_pred = transformer(embeds, embeds)[:,-1,:]
        y_pred = unembed(y_pred)
        loss = criterion(y_pred, y_train[batch])
        loss.backward()
        optim.step()
        losses.append(loss.item())
    print(loss.item())

X_gen = [X_test[:1,:,:]]
y_gen = []
for i in range(test_len):
    embeds = embed(X_gen[-1])
    y_pred = transformer(embeds, embeds)[:,-1,:]
    y_pred = unembed(y_pred).squeeze().detach().numpy()
    y_gen.append(y_pred.item())
    y_add = torch.tensor(y_pred).unsqueeze(0).unsqueeze(1).unsqueeze(-1)
    X_gen.append(torch.cat([X_gen[-1][:,1:,:], y_add], dim=1))
y_pred = np.array(y_gen)
print(f"Transformer: {time.time()-start_transformer}")

start_fft = time.time()
yf = np.fft.rfft(seq[:-test_len])
xt = X_seq[-test_len:]
signal = 0
freqs = []
amps = []
for i in range(len(yf)-1):
    n = len(yf)
    amp = yf[i] / n
    freq = (1/STEP)/2 * i / n
    if freq*(2*np.pi) > 3:
        break
    amps.append(np.abs(amp))
    freqs.append(freq * (2*np.pi))
    signal += np.real(amp) * np.cos(xt * freq * 2*np.pi)
    signal += np.imag(amp) * np.sin(xt * freq * 2*np.pi)
print(f"FFT: {time.time()-start_fft}")

plt.figure()
plt.plot(np.array(freqs), np.array(amps))
plt.xlabel("$2\pi \\times$ frequency")
plt.ylabel("Amplitude")

plt.figure()
plt.plot(np.arange(len(losses)), losses)

plt.figure()
plt.scatter(X_seq[:100], seq[:100])

plt.figure(figsize=(1.6*7,4.8))
y_test = y_test.squeeze().detach().numpy()
perfect_y = raw_seq[-test_len:]
plt.plot(xt[:500], y_test[:500], linewidth=2, label='True signal + noise')
plt.plot(xt[:500], signal[:500], color='tab:orange', linewidth=3, linestyle='--', label='FFT prediction')
plt.plot(xt[:500], y_pred[:500], color='tab:green', linewidth=3, linestyle='--', label='Transformer prediction')
# plt.plot(xt[:500], perfect_y[:500], color='black', linewidth=2)
plt.legend()

print("Transformer rMSE:", np.sqrt(np.mean((y_test - y_pred)**2)))
print("FFT rMSE:", np.sqrt(np.mean((y_test - signal)**2)))
print("Baseline rMSE:", np.sqrt(np.mean((y_test)**2)))
print("Perfect knowledge rMSE:", np.sqrt(np.mean((y_test - raw_seq[-test_len:])**2)))

plt.show()