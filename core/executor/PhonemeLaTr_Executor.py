import os
import sys
import json
from typing_extensions import override
import torch
import math
import pandas as pd
from torch.utils.data import DataLoader

from logger.logger import get_logger
from .base_executor import Base_Executor

from core.data import textlayout_ocr_adapt, PhonemeLaTrDataset
from core.tokenizer import PhonemeTokenizer

from timeit import default_timer as timer

import evaluation

from transformers import AutoTokenizer, AutoConfig
import itertools

log = get_logger(__name__)


class PhonemeLaTr_Executor(Base_Executor):
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        super().__init__(config, mode, evaltype, predicttype)
        log.info("---Initializing Executor---")

    def infer(self, dataloader, max_length):
        self.model.eval()

        decoded_preds = []

        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                pixel_values = batch['pixel_values'].to(self.config.DEVICE)
                coordinates = batch['coordinates'].to(self.config.DEVICE)
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE)
                ocr_attention_mask = batch['ocr_attention_mask'].to(self.config.DEVICE)
                tokenized_ocr = batch['tokenized_ocr'].to(self.config.DEVICE)

                pred = self.model.generate( pixel_values,
                                            coordinates,
                                            input_ids,
                                            src_attention_mask,
                                            ocr_attention_mask,
                                            tokenized_ocr,
                                            max_length = max_length,
                                            start_symbol = self.decode_tokenizer.bos_id,
                                            end_symbol = self.decode_tokenizer.eos_id,
                                            isgreedy = self.config.isgreedy,
                                            num_beam = self.config.num_beam)

                decoded_preds += self.decode_tokenizer.batch_decode(pred.tolist())

                log.info(f"|===| Inferring... {it+1} it |===|")

        return decoded_preds
    
    def _create_data_utils(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        train_qa_df = pd.read_csv(self.config.qa_train_path)[["image_id", "question", "answer", "filename"]]
        val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        self.val_answer = list(val_qa_df["answer"])

        self._create_decode_tokenizer()

        ocr_df = textlayout_ocr_adapt(self.config.ocr_path)

        log.info("# Creating Datasets")
        
        self.train_data = PhonemeLaTrDataset(base_img_path = self.config.base_img_path,
                                        qa_df = train_qa_df,
                                        ocr_df = ocr_df,
                                        tokenizer = self.tokenizer,
                                        decode_tokenizer = self.decode_tokenizer,
                                        max_ocr_element = self.config.max_ocr_element,
                                        max_ocr_length = self.config.max_ocr_length,
                                        transform=None,
                                        max_input_length = self.config.max_q_length,
                                        max_output_length = self.config.max_a_length)

        self.val_data = PhonemeLaTrDataset(base_img_path = self.config.base_img_path,
                                        qa_df = val_qa_df,
                                        ocr_df = ocr_df,
                                        tokenizer = self.tokenizer,
                                        decode_tokenizer = self.decode_tokenizer,
                                        max_ocr_element = self.config.max_ocr_element,
                                        max_ocr_length = self.config.max_ocr_length,
                                        transform=None,
                                        max_input_length = self.config.max_q_length,
                                        max_output_length = self.config.max_a_length)

    def _init_eval_predict_mode(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)
        self._create_decode_tokenizer()

        if self.mode == "eval":
            log.info("###Load eval data ...")
            val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.ocr_path)

            self.val_data = PhonemeLaTrDataset(base_img_path = self.config.base_img_path,
                                            qa_df = val_qa_df,
                                            ocr_df = ocr_df,
                                            tokenizer = self.tokenizer,
                                            decode_tokenizer = self.decode_tokenizer,
                                            max_ocr_element = self.config.max_ocr_element,
                                            max_ocr_length = self.config.max_ocr_length,
                                            transform=None,
                                            max_input_length = self.config.max_q_length,
                                            max_output_length = self.config.max_a_length)
            
            self.val_answer = list(val_qa_df["answer"])
            self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.EVAL_BATCH_SIZE)
            self.valiter_length = math.ceil(len(self.val_data)/self.config.EVAL_BATCH_SIZE)
        elif self.mode == "predict":
            log.info("###Load predict data ...")
            predict_qa_df = pd.read_csv(self.config.qa_predict_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.ocr_path)

            self.predict_data = PhonemeLaTrDataset(base_img_path = self.config.base_img_path,
                                                qa_df = predict_qa_df,
                                                ocr_df = ocr_df,
                                                tokenizer = self.tokenizer,
                                                decode_tokenizer = self.decode_tokenizer,
                                                max_ocr_element = self.config.max_ocr_element,
                                                max_ocr_length = self.config.max_ocr_length,
                                                transform=None,
                                                max_input_length = self.config.max_q_length,
                                                max_output_length = self.config.max_a_length)
            
            if self.config.get_predict_score:
                self.predict_answer = list(predict_qa_df["answer"])
            else:
                self.predict_answer = None

            self.predictiter = DataLoader(dataset = self.predict_data, 
                                    batch_size=self.config.PREDICT_BATCH_SIZE)

    def _train_epoch(self, epoch):
        self.model.train()
        losses = 0

        if epoch <= self.config.NUM_FREEZE_EPOCH:
            for child in self.model.encoder.children():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for child in self.model.encoder.children():
                for param in child.parameters():
                    param.requires_grad = True
        
        for it, batch in enumerate(self.trainiter):
            label_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
            labels = batch['label_ids'].type(torch.long).to(self.config.DEVICE)


            trg_input = labels[:, :-1]
            label_attention_mask = label_attention_mask[:, :-1]

            onset_logits, rhyme_logits, tone_logits= self.model(pixel_values = batch['pixel_values'].to(self.config.DEVICE),
                                                                coordinates = batch['coordinates'].to(self.config.DEVICE),
                                                                input_ids = batch['input_ids'].to(self.config.DEVICE),
                                                                labels = trg_input,
                                                                src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                                                label_attention_mask = label_attention_mask,
                                                                ocr_attention_mask=batch['ocr_attention_mask'].to(self.config.DEVICE) ,
                                                                tokenized_ocr=batch['tokenized_ocr'].to(self.config.DEVICE))


            self.optim.zero_grad()

            onset_tgt_out = labels[:, 1:, 0]
            rhyme_tgt_out = labels[:, 1:, 1]
            tone_tgt_out = labels[:, 1:, 2]


            onset_loss = self.loss_fn_onset(onset_logits.reshape(-1, onset_logits.shape[-1]), onset_tgt_out.reshape(-1))
            rhyme_loss = self.loss_fn_rhyme(rhyme_logits.reshape(-1, rhyme_logits.shape[-1]), rhyme_tgt_out.reshape(-1))
            tone_loss = self.loss_fn_tone(tone_logits.reshape(-1, tone_logits.shape[-1]), tone_tgt_out.reshape(-1))

            loss = onset_loss + rhyme_loss + tone_loss
            loss.backward()

            self.optim.step()

            self.scheduler.step()
            
            losses += loss.data.item()

            if it+1 == 1 or (it+1) % 20 == 0 or it+1==self.trainiter_length:
                log.info(f"--TRAINING--|Epoch: {epoch}| Step: {it+1}/{self.trainiter_length} | Loss: {round(losses / (it + 1), 2)}")

        return losses / self.trainiter_length
    
    def _evaluate(self):
        self.model.eval()
        losses = 0
        
        with torch.no_grad():
            for it, batch in enumerate(self.valiter):
                label_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
                labels = batch['label_ids'].type(torch.long).to(self.config.DEVICE)


                trg_input = labels[:, :-1]
                label_attention_mask = label_attention_mask[:, :-1]

                onset_logits, rhyme_logits, tone_logits= self.model(pixel_values = batch['pixel_values'].to(self.config.DEVICE),
                                                                coordinates = batch['coordinates'].to(self.config.DEVICE),
                                                                input_ids = batch['input_ids'].to(self.config.DEVICE),
                                                                labels = trg_input,
                                                                src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                                                label_attention_mask = label_attention_mask,
                                                                ocr_attention_mask=batch['ocr_attention_mask'].to(self.config.DEVICE) ,
                                                                tokenized_ocr=batch['tokenized_ocr'].to(self.config.DEVICE))

                onset_tgt_out = labels[:, 1:, 0]
                rhyme_tgt_out = labels[:, 1:, 1]
                tone_tgt_out = labels[:, 1:, 2]


                onset_loss = self.loss_fn_onset(onset_logits.reshape(-1, onset_logits.shape[-1]), onset_tgt_out.reshape(-1))
                rhyme_loss = self.loss_fn_rhyme(rhyme_logits.reshape(-1, rhyme_logits.shape[-1]), rhyme_tgt_out.reshape(-1))
                tone_loss = self.loss_fn_tone(tone_logits.reshape(-1, tone_logits.shape[-1]), tone_tgt_out.reshape(-1))
                
                loss = onset_loss + rhyme_loss + tone_loss

                losses += loss.data.item()

                if it+1 == 1 or (it+1) % 20 == 0 or it+1==self.valiter_length:
                    log.info(f"--VALIDATING--| Step: {it+1}/{self.valiter_length} | Loss: {round(losses / (it + 1), 2)}")


        return losses / self.valiter_length
    
    @override
    def _build_model(self):
        log.info(f"# Building model architecture ...")
        if self.config.MODEL_MOD_CONFIG_CLASS is not None:   
            self.model_config = self.build_class(self.config.MODEL_MOD_CONFIG_CLASS)().build(self.config)
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.backbone_name)

        self.model = self.build_class(self.config.MODEL_CLASS)(self.model_config, 
                                                                self.onset_vocab_size,
                                                                self.rhyme_vocab_size,
                                                                self.tone_vocab_size)
        self.model = self.model.to(self.config.DEVICE)
    
    @override
    def _init_training_properties(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.LR, betas=self.config.BETAS, eps=1e-9)
        self.loss_fn_onset = torch.nn.CrossEntropyLoss(ignore_index=self.decode_tokenizer.pad_id)    
        self.loss_fn_rhyme = torch.nn.CrossEntropyLoss(ignore_index=self.decode_tokenizer.pad_id)    
        self.loss_fn_tone = torch.nn.CrossEntropyLoss(ignore_index=self.decode_tokenizer.pad_id)    
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = self.optim, total_iters = self.config.warmup_step)

        self.SAVE = self.config.SAVE

        if os.path.isfile(os.path.join(self.config.SAVE_PATH, "last_ckp.pth")):
            log.info("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join(self.config.SAVE_PATH, "last_ckp.pth"))
            try:
                log.info(f"\t- Last train epoch: {ckp['epoch']}")
            except:
                log.info(f"\t- Last train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])
            self.optim.load_state_dict(ckp['optimizer'])
            self.scheduler.load_state_dict(ckp['scheduler'])
            self.best_score = ckp['best_score']
    
    def _create_decode_tokenizer(self):
        self.decode_tokenizer = PhonemeTokenizer(vocab_path=self.config.vocab_path,
                                                annotation_paths=self.config.annotation_paths)
        self.onset_vocab_size = len(self.decode_tokenizer.vocab['onset'])
        self.rhyme_vocab_size = len(self.decode_tokenizer.vocab['rhyme'])
        self.tone_vocab_size = len(self.decode_tokenizer.vocab['tone'])
