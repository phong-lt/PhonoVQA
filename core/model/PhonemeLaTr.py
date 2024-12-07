import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, ViTModel, AutoConfig
from .modules import SinusoidalPositionalEncoding, PhonemeEmbedding, BaseDecoder

class CustomizedLaTr_config:
    def build(self, config):
        model_config = AutoConfig.from_pretrained(config.encoder_name)
        model_config.update({
            "max_2d_position_embeddings": config.max_2d_position_embeddings,
            "vit_model": config.vit_model_name,
            "num_decoder_layers": config.num_decoder_layers,
            "n_head": config.n_head
        })
        return model_config

class SpatialModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_left_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.top_left_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.width_emb = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.height_emb = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)

    def forward(self, coordinates):
        top_left_x_feat = self.top_left_x(coordinates[:, :, 0])
        top_left_y_feat = self.top_left_y(coordinates[:, :, 1])
        bottom_right_x_feat = self.bottom_right_x(coordinates[:, :, 2])
        bottom_right_y_feat = self.bottom_right_y(coordinates[:, :, 3])
        width_feat = self.width_emb(coordinates[:, :, 4])
        height_feat = self.height_emb(coordinates[:, :, 5])

        layout_feature = (top_left_x_feat + top_left_y_feat +
                          bottom_right_x_feat + bottom_right_y_feat +
                          width_feat + height_feat)
        return layout_feature

class PhonemeLaTr(nn.Module):
    def __init__(self, 
                config, 
                onset_vocab_size,
                rhyme_vocab_size,
                tone_vocab_size):
        super().__init__()

        self.config = config
        self.encoder = T5EncoderModel.from_pretrained(
            self.config._name_or_path)

        self.spatial_feat_extractor = SpatialModule(config)
        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.visual_projector = nn.Linear(
            self.vit.config.hidden_size, self.encoder.config.d_model)

        # freeze ViT except the last dense layer
        for name, child in self.vit.named_children():
            for param in child.parameters():
                param.requires_grad = False

        ###### CUSTOM COMPONENTS ######
        self.rhyme_tone_embed_dim = self.encoder.config.d_model // 3
        self.onset_embed_dim = int(self.encoder.config.d_model - self.rhyme_tone_embed_dim*2)

        self.tgt_tok_emb = PhonemeEmbedding(
            onset_vocab_size,
            rhyme_vocab_size,
            tone_vocab_size,
            self.onset_embed_dim,
            self.rhyme_tone_embed_dim
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            self.encoder.config.d_model, dropout=0.1)

        self.decoder = BaseDecoder(
                            emb_size=self.encoder.config.d_model,
                            num_layers=self.config.num_decoder_layers,
                            n_head=self.config.n_head
                        )

        # shared lm_head
        self.shared_lm_head = nn.Linear(
            self.encoder.config.d_model, self.encoder.config.d_model)

        # Adjusted lm_heads to predict from decoder output directly
        self.onset_lm_head = nn.Linear(self.encoder.config.d_model, onset_vocab_size)
        self.rhyme_lm_head = nn.Linear(self.encoder.config.d_model, rhyme_vocab_size)
        self.tone_lm_head = nn.Linear(self.encoder.config.d_model, tone_vocab_size)

    def forward(self,
                pixel_values,
                coordinates,
                input_ids,
                labels,
                src_attention_mask,
                label_attention_mask,
                ocr_attention_mask,
                tokenized_ocr):

        inputs_embeds, attention_mask = self._calculate_embedding(
                pixel_values, coordinates, input_ids, ocr_attention_mask, src_attention_mask, tokenized_ocr)

        encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).last_hidden_state

        decoder_outputs = self.decode(labels, 
                                      encoder_outputs, 
                                      attention_mask, 
                                      label_attention_mask)

        decoder_outputs = self.shared_lm_head(decoder_outputs)  # (batch_size, seq_len, d_model)

        # Dự đoán đồng thời onset, rhyme, tone
        onset_logits = self.onset_lm_head(decoder_outputs)  # (batch_size, seq_len, onset_vocab_size)
        rhyme_logits = self.rhyme_lm_head(decoder_outputs)  # (batch_size, seq_len, rhyme_vocab_size)
        tone_logits = self.tone_lm_head(decoder_outputs)    # (batch_size, seq_len, tone_vocab_size)

        return onset_logits, rhyme_logits, tone_logits

    def decode(self, labels, encoder_outputs, encoder_attention_mask, label_attention_mask=None):
        square_subsequent_mask = self._create_square_subsequent_mask(labels.size(1), device=labels.device)
        
        label_embedding = self.positional_encoding(
                                self.tgt_tok_emb(labels))

        return self.decoder(label_embedding,
                            encoder_outputs,
                            tgt_mask = square_subsequent_mask,
                            memory_key_padding_mask = encoder_attention_mask,
                            tgt_key_padding_mask = label_attention_mask)

    def generate(self,
                 pixel_values,
                 coordinates,
                 input_ids,
                 src_attention_mask,
                 ocr_attention_mask,
                 tokenized_ocr,
                 start_symbol,
                 end_symbol,
                 max_length = 20,
                 isgreedy = True,
                 num_beam = 2):

        return self.greedy_generate(pixel_values,
                                    coordinates,
                                    input_ids,
                                    src_attention_mask,
                                    ocr_attention_mask,
                                    tokenized_ocr,
                                    start_symbol,
                                    end_symbol,
                                    max_length)

    def greedy_generate(self, 
                    pixel_values,
                    coordinates,
                    input_ids,
                    src_attention_mask,
                    ocr_attention_mask,
                    tokenized_ocr,
                    start_symbol,
                    end_symbol,
                    max_len=100):

        bz = input_ids.size(0)
        DEVICE = input_ids.device

        inputs_embeds, attention_mask = self._calculate_embedding(
                pixel_values, coordinates, input_ids, ocr_attention_mask, src_attention_mask, tokenized_ocr)

        encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).last_hidden_state

        # Khởi tạo với start_symbol (dạng [onset_start, rhyme_start, tone_start])
        ys = torch.tensor([[[start_symbol, 0, 0]]], dtype=torch.long).repeat(bz, 1, 1).to(DEVICE)

        for i in range(max_len):
            encoder_outputs = encoder_outputs.to(DEVICE)
            label_embedding = self.positional_encoding(
                                    self.tgt_tok_emb(ys))

            decoder_outputs = self.decoder(label_embedding,
                               encoder_outputs,
                               tgt_mask=self._create_square_subsequent_mask(ys.size(1), device=DEVICE),
                               memory_key_padding_mask=attention_mask)

            decoder_outputs = self.shared_lm_head(decoder_outputs)  # (batch_size, seq_len, d_model)

            # Lấy output cuối cùng
            last_decoder_output = decoder_outputs[:, -1, :]  # (batch_size, d_model)

            # Dự đoán đồng thời onset, rhyme, tone
            onset_logits = self.onset_lm_head(last_decoder_output)  # (batch_size, onset_vocab_size)
            onset_pred = torch.argmax(onset_logits, dim=-1)  # (batch_size)

            rhyme_logits = self.rhyme_lm_head(last_decoder_output)  # (batch_size, rhyme_vocab_size)
            rhyme_pred = torch.argmax(rhyme_logits, dim=-1)  # (batch_size)

            tone_logits = self.tone_lm_head(last_decoder_output)  # (batch_size, tone_vocab_size)
            tone_pred = torch.argmax(tone_logits, dim=-1)  # (batch_size)

            # Kết hợp các dự đoán
            next_word = torch.stack([onset_pred, rhyme_pred, tone_pred], dim=-1)  # (batch_size, 3)

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)  # (batch_size, seq_len + 1, 3)

            # Kiểm tra điều kiện dừng
            end_symbol_tensor = torch.tensor(end_symbol, dtype=torch.long, device=DEVICE)  # (3,)
            is_end = torch.all(next_word == end_symbol_tensor.unsqueeze(0), dim=-1)  # (batch_size)
            if torch.all(is_end):
                break

        return ys
            
    def _calculate_embedding(self, pixel_values, coordinates, input_ids, ocr_attention_mask, src_attention_mask, tokenized_ocr):
        img_feat = self.visual_projector(self.vit(pixel_values).last_hidden_state)
        spatial_feat = self.spatial_feat_extractor(coordinates)
        ocr_feat = self.encoder.shared(tokenized_ocr)
        language_feat = self.encoder.shared(input_ids)

        layout_feat = ocr_feat + spatial_feat

        multi_modal_feat = torch.cat([img_feat, layout_feat, language_feat], axis=1)
        input_attention_mask = torch.cat(
            [torch.ones(img_feat.shape[:2]).to(img_feat.device), ocr_attention_mask, src_attention_mask], axis=1)

        return multi_modal_feat, input_attention_mask

    def _create_square_subsequent_mask(self, sz, device="cuda"):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

