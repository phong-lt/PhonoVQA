import os
import sys
import json
import torch
import math
import pandas as pd
from torch.utils.data import DataLoader

from logger.logger import get_logger
from .base_executor import Base_Executor

from core.model import PreSTU, PreSTU_config
from core.data import textlayout_ocr_adapt, PreSTUDataset

from timeit import default_timer as timer

import evaluation

from transformers import AutoTokenizer, AutoConfig
import itertools

log = get_logger(__name__)


class PreSTU_Executor(Base_Executor):
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        super().__init__(config, mode, evaltype, predicttype)
        log.info("---Initializing Executor---")

    def infer(self, dataloader, max_length):
        self.model.eval()

        decoded_preds = []

        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                pixel_values = batch['pixel_values'].to(self.config.DEVICE)
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE)

                pred = self.model.generate( pixel_values,
                                            input_ids,
                                            src_attention_mask,
                                            max_length = max_length)

                decoded_preds += self.tokenizer.batch_decode(self._infer_post_processing(pred.tolist()), skip_special_tokens=True)

                log.info(f"|===| Inferring... {it+1} it |===|")

        return decoded_preds
    
    def _create_data_utils(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        train_qa_df = pd.read_csv(self.config.qa_train_path)[["image_id", "question", "answer", "filename"]]
        val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        self.val_answer = list(val_qa_df["answer"])

        ocr_df = textlayout_ocr_adapt(self.config.ocr_path)

        print("# Creating Datasets")
        
        self.train_data = PreSTUDataset(base_img_path = self.config.base_img_path,
                                        qa_df = train_qa_df,
                                        ocr_df = ocr_df,
                                        tokenizer = self.tokenizer,
                                        max_ocr_element = self.config.max_ocr_element,
                                        max_ocr_length = self.config.max_ocr_length,
                                        transform=None,
                                        max_input_length = self.config.max_q_length,
                                        max_output_length = self.config.max_a_length)

        self.val_data = PreSTUDataset(base_img_path = self.config.base_img_path,
                                        qa_df = val_qa_df,
                                        ocr_df = ocr_df,
                                        tokenizer = self.tokenizer,
                                        max_ocr_element = self.config.max_ocr_element,
                                        max_ocr_length = self.config.max_ocr_length,
                                        transform=None,
                                        max_input_length = self.config.max_q_length,
                                        max_output_length = self.config.max_a_length)

    def _init_eval_predict_mode(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        if self.mode == "eval":
            print("###Load eval data ...")
            val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.ocr_path)

            self.val_data = PreSTUDataset(base_img_path = self.config.base_img_path,
                                            qa_df = val_qa_df,
                                            ocr_df = ocr_df,
                                            tokenizer = self.tokenizer,
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
            print("###Load predict data ...")
            predict_qa_df = pd.read_csv(self.config.qa_predict_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.ocr_path)

            self.predict_data = PreSTUDataset(base_img_path = self.config.base_img_path,
                                                qa_df = predict_qa_df,
                                                ocr_df = ocr_df,
                                                tokenizer = self.tokenizer,
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
        
        for it, batch in enumerate(self.trainiter):
            label_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
            labels = batch['label_ids'].type(torch.long).to(self.config.DEVICE)


            trg_input = labels[:, :-1]
            label_attention_mask = label_attention_mask[:, :-1]

            logits = self.model(pixel_values = batch['pixel_values'].to(self.config.DEVICE),
                                input_ids = batch['input_ids'].to(self.config.DEVICE),
                                labels = trg_input,
                                src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                label_attention_mask = label_attention_mask,)


            self.optim.zero_grad()

            trg_out = labels[:, 1:]

            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
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

                logits = self.model(pixel_values = batch['pixel_values'].to(self.config.DEVICE),
                                    input_ids = batch['input_ids'].to(self.config.DEVICE),
                                    labels = trg_input,
                                    src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    label_attention_mask = label_attention_mask,)


                trg_out = labels[:, 1:]

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                losses += loss.data.item()

                if it+1 == 1 or (it+1) % 20 == 0 or it+1==self.valiter_length:
                    log.info(f"--VALIDATING--| Step: {it+1}/{self.valiter_length} | Loss: {round(losses / (it + 1), 2)}")


        return losses / self.valiter_length