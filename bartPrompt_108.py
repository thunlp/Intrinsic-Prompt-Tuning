import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration
from modeling_bart import BartForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right
from transformers.configuration_bart import BartConfig

from utils import label_smoothed_nll_loss

from transformers.modeling_bert import BertEncoder, BertConfig, BertLayerNorm

class MyBartPrompt_ensemble(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, prompt_num=100, ontology_idx_1=-1, ontology_idx_2=-1, ontology_general=-1, type1_num=25, type2_num=25, general_num=50):
        super().__init__(config)
        self.prompt_num = prompt_num
        self.d_model = config.d_model
        assert ontology_idx_1 > 0 and ontology_idx_2 > 0 and ontology_general > 0
        self.type1_num = type1_num
        self.type2_num = type2_num
        self.general_num = general_num
        self.ontology_idx_1 = ontology_idx_1
        self.ontology_idx_2 = ontology_idx_2
        self.ontology_general = ontology_general
        self.prompt_embeddings_ontology_type1 = nn.Embedding(self.ontology_idx_1 * self.type1_num, self.d_model)
        self.init_weights_prompt(self.prompt_embeddings_ontology_type1)
        self.prompt_embeddings_ontology_type2 = nn.Embedding(self.ontology_idx_2 * self.type2_num, self.d_model)
        self.init_weights_prompt(self.prompt_embeddings_ontology_type2)
        self.prompt_embeddings_general = nn.Embedding(self.ontology_general * self.general_num, self.d_model)
        self.init_weights_prompt(self.prompt_embeddings_general)

    def init_weights_prompt(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)

    def init_prompt(self, init_ids):
        self.model.encoder.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long))

    def generate_prompt(self, ontology):
        prompt_1 = self.prompt_embeddings_ontology_type1(ontology[0])
        prompt_2 = self.prompt_embeddings_ontology_type2(ontology[1])
        prompt_general = self.prompt_embeddings_general(ontology[2])
        return torch.cat([prompt_1, prompt_2, prompt_general], dim = 1)

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False, ontology=None, task_prompt_recored=None):
        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        else:
            _decoder_input_ids = decoder_input_ids
        
        if task_prompt_recored is None:
            task_prompt_recored = self.generate_prompt(ontology)
        else:
            task_prompt_recored = task_prompt_recored

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            task_prompt_recored=task_prompt_recored,
        )

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
            return loss
        return (lm_logits, ) + outputs[1:]

class MyBartPrompt_AE(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, prompt_num=100, intrinsic_dim=10, AE_loss=1, Distil_loss=1, alpha_AE=200, AE_type=0, AE_recover=False):
        super().__init__(config)
        self.prompt_num = prompt_num
        self.intrinsic_dim = intrinsic_dim
        self.d_model = config.d_model
        self.AE_loss = AE_loss
        self.Distil_loss = Distil_loss
        assert self.AE_loss + self.Distil_loss > 0, (self.AE_loss, self.Distil_loss)
        self.alpha_AE = alpha_AE
        self.AE_type = AE_type
        self.AE_recover = AE_recover
        if self.AE_type == 0:
            self.prompt_W1 = nn.Linear(prompt_num * config.d_model, intrinsic_dim)
            self.prompt_W2 = nn.Linear(intrinsic_dim, prompt_num * config.d_model)
            # self.init_weights_prompt(self.prompt_W1)
            # self.init_weights_prompt(self.prompt_W2)
        elif self.AE_type == 1:
            self.prompt_W1 = nn.Linear(prompt_num * config.d_model, intrinsic_dim)
            self.prompt_W2 = nn.Linear(intrinsic_dim, config.d_model)
            self.prompt_W3 = nn.Linear(config.d_model, prompt_num * config.d_model)
            # self.init_weights_prompt(self.prompt_W1)
            # self.init_weights_prompt(self.prompt_W2)
            # self.init_weights_prompt(self.prompt_W3)
        elif self.AE_type == 2:
            self.prompt_W1 = nn.Linear(prompt_num * config.d_model, 768)
            self.prompt_W2 = nn.Linear(768, intrinsic_dim)
            self.prompt_W3 = nn.Linear(intrinsic_dim, prompt_num * config.d_model)
            # self.init_weights_prompt(self.prompt_W1)
            # self.init_weights_prompt(self.prompt_W2)
            # self.init_weights_prompt(self.prompt_W3)
        elif self.AE_type == 3:
            self.prompt_W1 = nn.Linear(prompt_num * config.d_model, intrinsic_dim)
            self.prompt_W2 = nn.Linear(intrinsic_dim, prompt_num * config.d_model)
            # self.init_weights_prompt(self.prompt_W1)
            # self.init_weights_prompt(self.prompt_W2)
        elif self.AE_type == 4:
            self.prompt_W1 = nn.Linear(config.d_model, intrinsic_dim)
            self.prompt_W2 = nn.Linear(intrinsic_dim, 768)
            self.prompt_W3 = nn.Linear(768, config.d_model)
            # self.init_weights_prompt(self.prompt_W1)
            # self.init_weights_prompt(self.prompt_W2)
            # self.init_weights_prompt(self.prompt_W3)
        elif self.AE_type == 5:
            self.prompt_W1 = nn.Linear(prompt_num * config.d_model, intrinsic_dim)
            bertconfig = BertConfig(vocab_size = prompt_num, num_hidden_layers=1)
            self.prompt_W2 = nn.Linear(intrinsic_dim, prompt_num * config.d_model)
            self.prompt_bert = BertEncoder(bertconfig)
            self.prompt_position_embeddings = nn.Embedding(prompt_num, config.d_model)
            self.prompt_LayerNorm = BertLayerNorm(config.d_model, eps=1e-8)
        elif self.AE_type == 6:
            self.prompt_W1 = nn.Linear(prompt_num * config.d_model, intrinsic_dim)
            self.prompt_W2 = nn.Linear(intrinsic_dim, 768)
            self.prompt_W3 = nn.Linear(768, prompt_num * config.d_model)
            self.dropout = nn.Dropout(p=0.1)

        if self.AE_recover:
            self.prompt_task = nn.Embedding(1, intrinsic_dim)
            self.init_weights_prompt(self.prompt_task)

    def init_weights_prompt(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)

    def init_prompt(self, init_ids):
        self.model.encoder.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long))

    def generate_prompt(self, task_prompt):
        task_prompt_viewed = task_prompt.view(task_prompt.size()[0], self.d_model * self.prompt_num)
        if self.AE_type == 0:
            H1 = self.prompt_W1(task_prompt_viewed)
            H2 = self.prompt_W2(H1)
            task_prompt_recored = H2.view(task_prompt.size())
        elif self.AE_type == 1:
            H1 = self.prompt_W1(task_prompt_viewed)
            H2 = self.prompt_W2(H1)
            H3 = torch.tanh(H2)
            H4 = self.prompt_W3(H3)
            task_prompt_recored = H4.view(task_prompt.size())
        elif self.AE_type == 2:
            H1 = self.prompt_W1(task_prompt_viewed)
            H2 = torch.tanh(H1)
            H3 = self.prompt_W2(H2)
            H4 = self.prompt_W3(H3)
            task_prompt_recored = H4.view(task_prompt.size())
        elif self.AE_type == 3:
            H1 = self.prompt_W1(task_prompt_viewed)
            H2 = torch.tanh(H1)
            H3 = self.prompt_W2(H2)
            task_prompt_recored = H3.view(task_prompt.size())
        elif self.AE_type == 4:
            task_prompt_viewed = task_prompt_viewed.view(task_prompt.size())
            H1 = self.prompt_W1(task_prompt_viewed)
            H2 = self.prompt_W2(H1)
            H3 = torch.tanh(H2)
            H4 = self.prompt_W3(H3)
            task_prompt_recored = H4
        elif self.AE_type == 5:
            H1 = self.prompt_W1(task_prompt_viewed)
            H2 = self.prompt_W2(H1).view(task_prompt.size())
            H2 = H2 + self.prompt_position_embeddings.weight.unsqueeze(dim = 0).expand(task_prompt.size())
            H2 = self.prompt_LayerNorm(H2)
            H3 = self.prompt_bert(H2, head_mask=[None] * 1)
            task_prompt_recored = H3[0]
        elif self.AE_type == 6:
            H1 = self.prompt_W1(task_prompt_viewed)
            H2 = self.prompt_W2(H1)
            H3 = torch.tanh(H2)
            H3 = self.dropout(H3)
            H4 = self.prompt_W3(H3)
            task_prompt_recored = H4.view(task_prompt.size())

        return task_prompt_recored

    def generate_prompt_recover(self, task_prompt, bs):
        task_prompt_viewed = self.prompt_task.weight.expand(bs, -1)
        if self.AE_type == 0:
            H2 = self.prompt_W2(task_prompt_viewed)
            task_prompt_recored = H2.view(task_prompt.size())
        elif self.AE_type == 1:
            H2 = self.prompt_W2(task_prompt_viewed)
            H3 = torch.tanh(H2)
            H4 = self.prompt_W3(H3)
            task_prompt_recored = H4.view(task_prompt.size())
        elif self.AE_type == 2:
            H4 = self.prompt_W3(task_prompt_viewed)
            task_prompt_recored = H4.view(task_prompt.size())
        elif self.AE_type == 3:
            H3 = self.prompt_W2(task_prompt_viewed)
            task_prompt_recored = H3.view(task_prompt.size())
        elif self.AE_type == 5:
            H2 = self.prompt_W2(task_prompt_viewed).view(task_prompt.size())
            H2 = H2 + self.prompt_position_embeddings.weight.unsqueeze(dim = 0).expand(task_prompt.size())
            H2 = self.prompt_LayerNorm(H2)
            H3 = self.prompt_bert(H2, head_mask=[None] * 1)
            task_prompt_recored = H3[0]
        elif self.AE_type == 6:
            H2 = self.prompt_W2(task_prompt_viewed)
            H3 = torch.tanh(H2)
            H4 = self.prompt_W3(H3)
            task_prompt_recored = H4.view(task_prompt.size())
        return task_prompt_recored

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False, task_prompt=None, task_prompt_recored=None):
        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        else:
            _decoder_input_ids = decoder_input_ids            

        if task_prompt_recored is None:
            if self.AE_recover:
                task_prompt_recored = self.generate_prompt_recover(task_prompt, input_ids.size()[0])
            else:
                task_prompt_recored = self.generate_prompt(task_prompt)

        if is_training and self.Distil_loss == 0:
            pass    
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=_decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache,
                task_prompt_recored=task_prompt_recored,
            )

            lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        
        if is_training:
            loss = 0
            if self.Distil_loss == 1:
                lprobs = F.log_softmax(lm_logits, dim=-1)
                loss_distil, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
                loss += loss_distil
            else:
                loss_distil = 0
            if self.AE_loss == 1:
                loss_AE = torch.mean((task_prompt_recored - task_prompt)**2) * self.alpha_AE
                loss += loss_AE
            else:
                loss_AE = 0
            loss = loss_distil + loss_AE
            # only for supporting yard env, loss_distil and loss_AE is set to 0 in outer function!
            if self.Distil_loss == 0:
                loss_distil = loss
            if self.AE_loss == 0:
                loss_AE = loss
            return loss, loss_distil, loss_AE
        return (lm_logits, ) + outputs[1:]

class MyBartPrompt(BartForConditionalGeneration):

    def init_prompt(self, init_ids):
        self.model.encoder.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long))

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):
        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id)
            return loss
        return (lm_logits, ) + outputs[1:]


