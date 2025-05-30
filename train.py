import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from data import get_batched_dataset, get_vocab_sizes, get_word_index
from model import Seq2Seq, Encoder, Decoder
from data import tokenize_with_punctuation
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
    
# Get dataset and vocabulary sizes
batched_dataset = get_batched_dataset()
F_vocab_size, E_vocab_size = get_vocab_sizes()
E_w_to_i, E_i_to_w, F_w_to_i, F_i_to_w = get_word_index()

# Model hyperparameters
embedding_size = 200
hidden_size = 200
num_layers = 2
dropout = 0.5
learning_rate = 0.01
epochs = 7

# Initialize model
encoder = Encoder(E_vocab_size, embedding_size, hidden_size, num_layers, dropout)
decoder = Decoder(F_vocab_size, embedding_size, hidden_size, num_layers, dropout)
model = Seq2Seq(encoder, decoder)
model = model.to(device)

def prepare_sentence(sentence):
    """Prepare a sentence for translation by tokenizing and converting to tensor."""
    tokens = tokenize_with_punctuation(sentence)
    tokens = [E_w_to_i['<sos>']] + [E_w_to_i.get(word, E_w_to_i['<pad>']) for word in tokens] + [E_w_to_i['<eos>']]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension


def train(model, epochs, lr, test_sentence="I don't have any rooms for rent."):
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    best_loss = float('inf')
    
    # Create directory for model checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in batched_dataset:
            X, y = batch
            # print(f"Batch size X: {X.size(0)}, y: {y.size(0)}")
            
            optimizer.zero_grad()

            # Move both input and target to the same device
            X = X.to(device)
            y = y.to(device)
            
            output = model(X, y)
            # Ensure both output and target are on the same device before loss calculation
            output = output.to(device)
            y = y.to(device)
            
            loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"New best model saved with loss: {best_loss:.4f}")
        
        # Translate test sentence
        model.eval()
        with torch.no_grad():
            test_input = prepare_sentence(test_sentence)
            test_input = test_input.to(device)
            
            # Get encoder output
            hn, cn = model.encoder(test_input)

            
            # Initialize decoder input with <sos> token
            decoder_input = torch.tensor([F_w_to_i['<sos>']], device=device).unsqueeze(0)  # shape: [1, 1]
            print(decoder_input.shape)
            # Generate translation
            translation = []
            for _ in range(50):  # Maximum length of translation

                output, hn, cn = model.decoder(decoder_input, hn, cn)
                
                predicted_token = output.argmax(dim=-1)
                translation.append(predicted_token.item())
                
                print("Input shape:", output.shape)


                # Stop if we predict <eos>
                if predicted_token.item() == F_w_to_i['<eos>']:
                    break
                    
                decoder_input = predicted_token.unsqueeze(0)
            
            # Convert indices back to words
            translation_words = [F_i_to_w[idx] for idx in translation]
            translation_text = ' '.join(translation_words)
            print(f"Test translation: {test_sentence} -> {translation_text}")
            
        
        print("-" * 50)

if __name__ == "__main__":
    train(model, epochs, learning_rate)
