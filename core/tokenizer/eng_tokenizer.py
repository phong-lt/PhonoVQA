import string

class CharTokenizerEn:
    def __init__(self,
                 pad_token="<pad>",
                 bos_token="<bos>",
                 eos_token="<eos>",
                 unk_token="<unk>"):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.special_tokens = [pad_token, bos_token, eos_token, unk_token]
        self.idx2str = list(string.printable) + self.special_tokens
        self.str2idx = {char: idx for idx, char in enumerate(self.idx2str)}
        self.pad_id = self.str2idx[self.pad_token]
        self.bos_id = self.str2idx[self.bos_token]
        self.eos_id = self.str2idx[self.eos_token]
        self.unk_id = self.str2idx[self.unk_token]

    def __call__(self, text, max_length=None, padding=True, add_special_tokens=True):
        if isinstance(text, list):
            return self.batch_encode(text, max_length, padding, add_special_tokens)
        return self.encode(text, max_length, padding, add_special_tokens)

    def __len__(self):
        return len(self.idx2str)

    def encode(self, text, max_length=None, padding=True, add_special_tokens=True):
        char_seq = list(text)
        for i in range(len(char_seq)):
            char_seq[i] = self.str2idx.get(char_seq[i], self.unk_id)
        length = len(char_seq) + (2 if add_special_tokens else 0)
        if max_length is None:
            max_length = length
        if length > max_length:
            limit = max_length - (2 if add_special_tokens else 0)
            char_seq = char_seq[:limit]
            length = max_length
        if add_special_tokens:
            encoding = self.add_special_tokens(char_seq, length, max_length, padding)
        else:
            encoding = char_seq
            if padding and length < max_length:
                encoding += [self.pad_id] * (max_length - length)
        return encoding

    def add_special_tokens(self, encoding, length, max_len, padding):
        encoding = [self.bos_id] + encoding + [self.eos_id]
        if padding:
            cur_len = len(encoding)
            if cur_len < max_len:
                encoding += [self.pad_id] * (max_len - cur_len)
        return encoding

    def batch_encode(self, texts, max_length=None, padding=True, add_special_tokens=True):
        return [self.encode(text, max_length, padding, add_special_tokens) for text in texts]

    def post_processing(self, out_ids):
        eos_token_id = self.str2idx[self.eos_token]
        results = []
        for seq in out_ids:
            if eos_token_id in seq:
                eos_pos = seq.index(eos_token_id)
                results.append(seq[1:eos_pos])
            else:
                if seq and seq[0] == self.bos_id:
                    results.append(seq[1:])
                else:
                    results.append(seq)
        return results

    def decode(self, char_seq):
        seq_processed = self.post_processing([char_seq])[0]
        text = "".join(
            self.idx2str[token_id]
            for token_id in seq_processed
            if token_id < len(self.idx2str) and self.idx2str[token_id] not in self.special_tokens
        )
        return text

    def batch_decode(self, char_seqs):
        seqs_processed = self.post_processing(char_seqs)
        texts = []
        for seq in seqs_processed:
            text = "".join(
                self.idx2str[token_id]
                for token_id in seq
                if token_id < len(self.idx2str) and self.idx2str[token_id] not in self.special_tokens
            )
            texts.append(text)
        return texts
