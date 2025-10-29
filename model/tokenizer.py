import pickle

class SimpleTokenizer:
    def __init__(self, tokenizer_path='saved_models/tokenizer_simple.pkl'):
        # Charger les données du tokenizer
        with open(tokenizer_path, 'rb') as f:
            data = pickle.load(f)
        
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = data['idx_to_char']
        self.pad_token = data['special_tokens']['pad_token']
        self.sos_token = data['special_tokens']['sos_token']
        self.eos_token = data['special_tokens']['eos_token']
        self.unk_token = data['special_tokens']['unk_token']
        self.pad_token_id = data['token_ids']['pad_token_id']
        self.sos_token_id = data['token_ids']['sos_token_id']
        self.eos_token_id = data['token_ids']['eos_token_id']
        self.unk_token_id = data['token_ids']['unk_token_id']
        self.chars = data['chars']
        self.vocab_size = data['vocab_size']
        self.vocab = list(self.idx_to_char.values())

    def encode(self, text, add_special_tokens=True):
        """Transforme texte → nombres"""
        tokens = [self.char_to_idx.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            tokens = [self.sos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, tokens):
        """Transforme nombres → texte"""
        chars = [self.idx_to_char.get(t, self.unk_token) for t in tokens
                if t not in [self.pad_token_id, self.sos_token_id, self.eos_token_id]]
        return ''.join(chars)