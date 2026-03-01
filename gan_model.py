import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
generator = BartForConditionalGeneration.from_pretrained(model_name).to(device)
def generate_summary(article, max_len=120):
    inputs = tokenizer(
        article,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    ids = generator.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=int(max_len * 0.4),
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = tokenizer.decode(ids[0], skip_special_tokens=True)
    return summary, ids
class ConditionalDiscriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, article_ids, summary_ids):
        combined = torch.cat((article_ids, summary_ids), dim=1)
        x = self.embedding(combined)
        x = torch.mean(x, dim=1)
        return self.network(x)
discriminator = ConditionalDiscriminator(tokenizer.vocab_size).to(device)