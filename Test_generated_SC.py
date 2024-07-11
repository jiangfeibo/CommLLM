import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import re
# prepare training data
def get_data():
    data = []
    max_words = 0
    with open("movie_lines.txt","r",encoding="utf-8")as f:
        content = f.read()
        req = ".*?\+\+\+\$\+\+\+.*?\+\+\+\$\+\+\+.*?\+\+\+\$\+\+\+.*?\+\+\+\$\+\+\+ "
        lines = re.split(req,content)
        print(len(lines))
        for i,line in enumerate(lines):
            line = line.replace("\n","").strip()
            req = "\.|\?"
            sentence = re.split(req,line)
            sentence = set(sentence)
            if '' in sentence:
                sentence.remove('')
            if ' ' in sentence:
                sentence.remove(' ')
            data+=list(sentence)
    new_data = []
    for sen in data:
        if len(sen.split(" ")) > 20 or len(sen.split(" ")) < 2:
            continue
        else:
            new_data.append(sen)
        max_words = max(len(sen.split(" ")),max_words)
    return new_data[:10000]

training_data = get_data()

# Tokenize sentences into words
words = [sentence.split() for sentence in training_data]

# Create a vocabulary
vocab = list(set(word for sentence in words for word in sentence))
vocab_size = 8266
print(vocab_size)

# Convert words to unique indices
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Convert sentences to numerical representation
numerical_sentences = [[word_to_idx[word] for word in sentence] for sentence in words]
# Spilt training and test data
data_size = len(numerical_sentences)
train_data = numerical_sentences[:int(data_size*0.8)]
test_data = numerical_sentences[int(data_size*0.8):]

## Copy the generated code of the SC model here
# Semantic Encoder
class SemanticEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(SemanticEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        print(embedded.shape)
        output, (hidden, cell) = self.lstm(embedded)
        return output

# Channel Encoder and Decoder (Simple Identity Mapping)
class ChannelEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(ChannelEncoder, self).__init__()
        self.identity = nn.Identity()

    def forward(self, input_features):
        return self.identity(input_features)

class ChannelDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(ChannelDecoder, self).__init__()
        self.identity = nn.Identity()

    def forward(self, received_features):
        return self.identity(received_features)

# Define the physical channel, which is a Gaussian white noise channel with a given SNR
class PhysicalChannel(nn.Module):
    def __init__(self, snr):
        super(PhysicalChannel, self).__init__()
        self.snr = snr

    def forward(self, x):
        x = x.cpu()
        # x: (batch_size, output_size)
        noise_power = 10 ** (-self.snr / 10) # Calculate the noise power from the SNR
        noise = math.sqrt(noise_power) * torch.randn_like(x)  # Generate Gaussian white noise with the same shape as x
        y = x + noise  # Add noise to the signal
        y = y.to(device)
        return y

# Semantic Decoder
class SemanticDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(SemanticDecoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden):
        output, _ = self.lstm(hidden)
        output = self.linear(output)
        return output

class SC_model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(SC_model, self).__init__()
        self.semantic_encoder = SemanticEncoder(vocab_size, embedding_size, hidden_size)
        self.channel_encoder = ChannelEncoder(hidden_size)
        self.channel_decoder = ChannelDecoder(hidden_size)
        self.semantic_decoder = SemanticDecoder(hidden_size, vocab_size)
        # self.physical_channel = PhysicalChannel(snr)

    def forward(self,x):
        x = self.semantic_encoder(x)
        x = self.channel_encoder(x)
        # x = self.physical_channel(x)
        x = self.channel_decoder(x)
        x = self.semantic_decoder(x)
        return x

# training semantic communication model
def train():
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        for sentence in train_data:
            if sentence == []:
                continue
            optimizer.zero_grad()
            input_seq = torch.tensor(sentence).to(device).long()  # Input: all words except the last
            target_seq = torch.tensor(sentence).to(device).long()  # Target: all words except the first

            output = model(input_seq)
            print(output.shape, target_seq.shape)

            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')
        torch.save(model.state_dict(), f"{snr}_weight.pth")
        eval()

# evaluate semantic communication model
def eval():
    # Test the semantic communication model
    scores = []
    model.eval()
    model.load_state_dict(torch.load(f"{snr}_weight.pth", map_location="cpu"))
    for i, sentence in enumerate(test_data):
        try:
            if sentence == []:
                continue
            test_input = torch.tensor(sentence).to(device)
            with torch.no_grad():
                output = model(test_input)
                predicted_indices = torch.argmax(output, dim=1).cpu().numpy()
                predicted_sentence = ' '.join([idx_to_word[idx] for idx in predicted_indices])
                src_txt = training_data[i + int(data_size * 0.8)]
                tar_txt = predicted_sentence
                # print("Original Sentence:", src_txt)
                # print("Predicted Sentence:", tar_txt)

                # Tokenize and process each sentence individually
                encoded_sentence1 = tokenizer.encode_plus(src_txt, add_special_tokens=True, max_length=64,
                                                          truncation=True, return_tensors='pt', padding='max_length')
                encoded_sentence2 = tokenizer.encode_plus(tar_txt, add_special_tokens=True, max_length=64,
                                                          truncation=True, return_tensors='pt', padding='max_length')

                # Obtain the BERT embeddings for each sentence

                model_output1 = bert(encoded_sentence1['input_ids'], encoded_sentence1['attention_mask'])
                embeddings1 = model_output1.last_hidden_state[:, 0, :]

                model_output2 = bert(encoded_sentence2['input_ids'], encoded_sentence2['attention_mask'])
                embeddings2 = model_output2.last_hidden_state[:, 0, :]

                # Calculate the similarity using cosine similarity
                similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
                print(f"Cosine similarity score: {similarity}")
                scores.append(similarity)
        except Exception as e:
            print(e)
            pass

    print("SNR:", snr, "sim score:", np.mean(scores))

if __name__ == '__main__':
    # Instantiate the model components
    from torchsummary import summary
    from transformers import BertTokenizer, BertModel
    from sklearn.metrics.pairwise import cosine_similarity

    # Load the BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('Geotrend/bert-base-en-bg-cased')
    bert = BertModel.from_pretrained('Geotrend/bert-base-en-bg-cased')
    embedding_size = 64
    hidden_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SC_model(vocab_size, embedding_size, hidden_size).to(device)
    for snr in reversed([15,10,5,0,-5]):
        train()



