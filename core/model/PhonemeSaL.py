import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import T5LayerNorm

from .modules import (
    T52DEncoderModel, 
    RelativePositionBias1D, 
    SCPRelativePositionBias,
    RelativePositionBiasAggregated,
    SinusoidalPositionalEncoding, 
    PhonemeEmbedding, 
    BaseDecoder 
)

class CustomizedSaL_config:
    def build(self, config, new_token_embedding_size):
        model_config = AutoConfig.from_pretrained(config.backbone_name)

        model_config.update({"ocr_hidden" : config.ocr_hidden,  
                                "obj_hidden" : config.obj_hidden,
                                "new_token_embedding_size": new_token_embedding_size,
                                "num_decoder_layers": config.num_decoder_layers,
                                "n_head": config.n_head})
        
        return model_config


class PhonemeSaL(nn.Module):
    def __init__(self, 
                config, 
                onset_vocab_size,
                rhyme_vocab_size,
                tone_vocab_size, 
                obj_dropout=0.1, 
                ocr_dropout=0.1):
        super().__init__()

        self.config = config
        self.encoder = T52DEncoderModel.from_pretrained(self.config._name_or_path)
        self.encoder.resize_token_embeddings(self.config.new_token_embedding_size)
        
        self.rel2Dbias = RelativePositionBiasAggregated(Relative1D=RelativePositionBias1D(num_heads = self.encoder.config.num_heads),
                                                        SCP=SCPRelativePositionBias(num_heads = self.encoder.config.num_heads))

        self.obj_dropout = nn.Dropout(obj_dropout)
        self.obj_feature_projector = nn.Linear(self.config.obj_hidden, self.encoder.config.d_model)
        self.obj_bbox_projector = nn.Linear(4, self.encoder.config.d_model)
        self.obj_feature_layer_norm = T5LayerNorm(self.encoder.config.d_model)

        self.ocr_dropout = nn.Dropout(ocr_dropout)
        self.ocr_feature_projector = nn.Linear(self.config.ocr_hidden, self.encoder.config.d_model)
        self.ocr_bbox_projector = nn.Linear(4, self.encoder.config.d_model)
        self.ocr_feature_layer_norm = T5LayerNorm(self.encoder.config.d_model)

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
                input_ids,
                src_attention_mask,
                label_ids,
                label_attention_mask,
                tokenized_ocr,
                ocr_attention_mask,
                ocr_coordinates,
                ocr_features,
                tokenized_obj,
                obj_attention_mask,
                obj_coordinates,
                obj_features,
                max_ocr,
                max_ques) :

        obj_inputs_embeds = self._calculate_obj_embedding(
                tokenized_obj, obj_coordinates, obj_features)
        
        ocr_inputs_embeds = self._calculate_ocr_embedding(
                tokenized_ocr, ocr_coordinates, ocr_features)
        
        ques_inputs_embeds = self.encoder.shared(input_ids)

        multi_modal_feat = torch.cat([ques_inputs_embeds, ocr_inputs_embeds, obj_inputs_embeds], dim=1)

        input_attention_mask = torch.cat(
            [src_attention_mask, ocr_attention_mask, obj_attention_mask], dim=1)
        
        position_bias = self.rel2Dbias(multi_modal_feat, input_attention_mask, ocr_coordinates, max_ques, max_ocr)

        encoder_outputs = self.encoder(
                attention_mask=input_attention_mask,
                inputs_embeds=multi_modal_feat,
                position_bias=position_bias
            ).last_hidden_state

        decoder_outputs = self.decode(label_ids, 
                                        encoder_outputs, 
                                        input_attention_mask, 
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
                input_ids,
                src_attention_mask,
                tokenized_ocr,
                ocr_attention_mask,
                ocr_coordinates,
                ocr_features,
                tokenized_obj,
                obj_attention_mask,
                obj_coordinates,
                obj_features,
                max_ocr,
                max_ques,
                start_symbol,
                end_symbol,
                max_length = 20,
                isgreedy = True,
                num_beam = 2):

        
        return self.greedy_generate(input_ids,
                                    src_attention_mask,
                                    tokenized_ocr,
                                    ocr_attention_mask,
                                    ocr_coordinates,
                                    ocr_features,
                                    tokenized_obj,
                                    obj_attention_mask,
                                    obj_coordinates,
                                    obj_features,
                                    max_ocr,
                                    max_ques,
                                    start_symbol,
                                    end_symbol,
                                    max_length)

    
    def greedy_generate(self, 
                    input_ids,
                    src_attention_mask,
                    tokenized_ocr,
                    ocr_attention_mask,
                    ocr_coordinates,
                    ocr_features,
                    tokenized_obj,
                    obj_attention_mask,
                    obj_coordinates,
                    obj_features,
                    max_ocr,
                    max_ques,
                    start_symbol,
                    end_symbol,
                    max_len=100):
        
        
        bz = input_ids.size(0)
        DEVICE = input_ids.device

        obj_inputs_embeds = self._calculate_obj_embedding(
                tokenized_obj, obj_coordinates, obj_features)
        
        ocr_inputs_embeds = self._calculate_ocr_embedding(
                tokenized_ocr, ocr_coordinates, ocr_features)
        
        ques_inputs_embeds = self.encoder.shared(input_ids)

        multi_modal_feat = torch.cat([ques_inputs_embeds, ocr_inputs_embeds, obj_inputs_embeds], dim=1)

        input_attention_mask = torch.cat(
            [src_attention_mask, ocr_attention_mask, obj_attention_mask], dim=1)
        
        position_bias = self.rel2Dbias(multi_modal_feat, input_attention_mask, ocr_coordinates, max_ques, max_ocr)

        encoder_outputs = self.encoder(
                attention_mask=input_attention_mask,
                inputs_embeds=multi_modal_feat,
                position_bias=position_bias
            ).last_hidden_state

        ys = torch.tensor([[[start_symbol, 0, 0]]], dtype=torch.long).repeat(bz, 1, 1).to(DEVICE)

        for i in range(max_len):
            encoder_outputs = encoder_outputs.to(DEVICE)

            out = self.decode(ys, encoder_outputs, input_attention_mask)
            
            decoder_outputs = self.shared_lm_head(out)  # (batch_size, seq_len, d_model)

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
    
    
    def _calculate_obj_embedding(self, tokenized_obj, obj_coordinates, obj_features):
        return self.obj_feature_layer_norm(self.obj_feature_projector(obj_features))\
            + self.obj_feature_layer_norm(self.obj_bbox_projector(obj_coordinates))\
            + self.encoder.shared(tokenized_obj)
    
    def _calculate_ocr_embedding(self, tokenized_ocr, ocr_coordinates, ocr_features):
        return self.ocr_feature_layer_norm(self.ocr_feature_projector(ocr_features))\
            + self.ocr_feature_layer_norm(self.ocr_bbox_projector(ocr_coordinates))\
            + self.encoder.shared(tokenized_ocr)
    
    def _create_square_subsequent_mask(self, sz, device="cuda"):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask