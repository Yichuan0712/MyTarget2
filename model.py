from peft import get_peft_model
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit
import torch
from torch import nn

torch.manual_seed(0)
import os
from transformers import AutoTokenizer
from PromptProtein.utils import PromptConverter
from PromptProtein.models import openprotein_promptprotein
from transformers import EsmModel
from utils import customlog, prepare_saving_dir
from train import make_buffer


class LayerNormNet(nn.Module):
    """
    From https://github.com/tttianhao/CLEAN
    """

    def __init__(self, configs, hidden_dim=512, out_dim=256):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = configs.supcon.drop_out
        self.device = configs.train_settings.device
        self.dtype = torch.float32
        feature_dim = {"facebook/esm2_t6_8M_UR50D": 320, "facebook/esm2_t33_650M_UR50D": 1280,
                       "facebook/esm2_t30_150M_UR50D": 640, "facebook/esm2_t12_35M_UR50D": 480}
        self.fc1 = nn.Linear(feature_dim[configs.encoder.model_name], hidden_dim, dtype=self.dtype, device=self.device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=self.dtype, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=self.dtype, device=self.device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=self.dtype, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=self.dtype, device=self.device)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        # x = self.dropout(self.ln2(self.fc2(x)))
        # x = torch.relu(x)
        x = self.fc3(x)
        return x


def initialize_PromptProtein(pretrain_loc, trainable_layers):
    from PromptProtein.models import openprotein_promptprotein
    from PromptProtein.utils import PromptConverter
    model, dictionary = openprotein_promptprotein(pretrain_loc)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        for lname in trainable_layers:
            if lname in name:
                param.requires_grad = True
    return model


class CustomPromptModel(nn.Module):
    def __init__(self, configs, pretrain_loc, trainable_layers):
        super(CustomPromptModel, self).__init__()
        self.pretrain = initialize_PromptProtein(pretrain_loc, trainable_layers)
        # self.learnable_prompt_emb = nn.Parameter(initialize_learnable_prompts(prompt_tok_vec, self.pretrain, self.converter)).to("cuda")
        # self.learnable_prompt_emb = nn.Parameter(initialize_learnable_prompts(8, self.pretrain, self.converter)).to("cuda")
        # self.learnable_prompt_emb = nn.Parameter(initialize_learnable_prompts(8, self.pretrain, self.converter, self.device)).to(self.device)
        # self.decoder_class = Decoder_linear(input_dim=1280, output_dim=5)
        self.pooling_layer = nn.AdaptiveAvgPool1d(output_size=1)
        self.head = nn.Linear(1280, configs.encoder.num_classes)
        self.cs_head = nn.Linear(1280, 1)
        # if decoder_cs=="linear":
        #     self.decoder_cs = Decoder_linear(input_dim=1280, output_dim=1)

    def forward(self, encoded_seq):
        result = self.pretrain(encoded_seq, with_prompt_num=1)
        # logits size => (B, T+2, E)
        logits = result['logits']

        transposed_feature = logits.transpose(1, 2)
        pooled_features = self.pooling_layer(transposed_feature).squeeze(2)
        type_probab = self.head(pooled_features)
        cs_head = self.cs_head(logits).squeeze(dim=-1)
        cs_pred = cs_head[:, 1:]
        # print("peptide_probab size ="+str(type_probab.size())) 
        return type_probab, cs_pred


def prepare_tokenizer(configs, curdir_path):
    if configs.encoder.composition == "esm_v2":
        tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)
    elif configs.encoder.composition == "promprot":
        model, dictionary = openprotein_promptprotein(os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"))
        tokenizer = PromptConverter(dictionary)
    elif configs.encoder.composition == "both":
        tokenizer_esm = AutoTokenizer.from_pretrained(configs.encoder.model_name)
        model, dictionary = openprotein_promptprotein(os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"))
        tokenizer_promprot = PromptConverter(dictionary)
        tokenizer = {"tokenizer_esm": tokenizer_esm, "tokenizer_promprot": tokenizer_promprot}
    return tokenizer


def tokenize(tools, seq):
    if tools['composition'] == "esm_v2":
        max_length = tools['max_len']
        encoded_sequence = tools["tokenizer"](seq, max_length=max_length, padding='max_length',
                                              truncation=True,
                                              return_tensors="pt"
                                              )
        # encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
        # encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])
    elif tools['composition'] == "promprot":
        if tools['prm4prmpro'] == 'seq':
            prompts = ['<seq>']
            encoded_sequence = tools["tokenizer"](seq, prompt_toks=prompts)
        elif tools['prm4prmpro'] == 'ppi':
            prompts = ['<ppi>']
            encoded_sequence = tools["tokenizer"](seq, prompt_toks=prompts)
    elif tools['composition'] == "both":
        max_length = tools['max_len']
        encoded_sequence_esm2 = tools["tokenizer"]["tokenizer_esm"](seq, max_length=max_length, padding='max_length',
                                                                    truncation=True,
                                                                    return_tensors="pt"
                                                                    )
        if tools['prm4prmpro'] == 'seq':
            prompts = ['<seq>']
            encoded_sequence_promprot = tools["tokenizer"]["tokenizer_promprot"](seq, prompt_toks=prompts)
        elif tools['prm4prmpro'] == 'ppi':
            prompts = ['<ppi>']
            encoded_sequence_promprot = tools["tokenizer"]["tokenizer_promprot"](seq, prompt_toks=prompts)
        encoded_sequence = {"encoded_sequence_esm2": encoded_sequence_esm2,
                            "encoded_sequence_promprot": encoded_sequence_promprot}
    return encoded_sequence


def print_trainable_parameters(model, logfilepath):
    """
        Prints the number of trainable parameters in the model.
        """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    customlog(logfilepath,
              f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}\n")


def prepare_esm_model(model_name, configs):
    model = EsmModel.from_pretrained(model_name)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    if configs.PEFT == "lora":
        config = LoraConfig(target_modules=["query", "key"])
        model = get_peft_model(model, config)
    # elif configs.PEFT == "PromT":
    #     config = PromptTuningConfig(task_type="SEQ_CLS", prompt_tuning_init=PromptTuningInit.TEXT, num_virtual_tokens=8, 
    #                             prompt_tuning_init_text="Classify what the peptide type of a protein sequence", tokenizer_name_or_path=configs.encoder.model_name)
    #     model = get_peft_model(model, config)
    #     for param in model.encoder.layer[-1].parameters():
    #         param.requires_grad = True
    #     for param in model.pooler.parameters():
    #         param.requires_grad = False
    elif configs.PEFT == "frozen":
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
    elif configs.PEFT == "PFT":
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.encoder.layer[configs.train_settings.fine_tune_lr:].parameters():
            param.requires_grad = True
        for param in model.pooler.parameters():
            param.requires_grad = False
    elif configs.PEFT == "lora_PFT":
        config = LoraConfig(target_modules=["query", "key"])
        model = get_peft_model(model, config)
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.encoder.layer[configs.train_settings.fine_tune_lr:].parameters():
            param.requires_grad = True
        for param in model.pooler.parameters():
            param.requires_grad = False
    return model


class ParallelLinearDecoders(nn.Module):
    def __init__(self, input_size, output_sizes):
        super(ParallelLinearDecoders, self).__init__()
        self.linear_decoders = nn.ModuleList([
            nn.Linear(input_size, output_size) for output_size in output_sizes
        ])

    def forward(self, x):
        decoder_outputs = [decoder(x) for decoder in self.linear_decoders]
        return decoder_outputs


def remove_s_e_token(target_tensor, mask_tensor):  # target_tensor [batch, seq+2, ...]  =>  [batch, seq, ...]
    # mask_tensor=inputs['attention_mask']
    # input_tensor=inputs['input_ids']
    result = []
    for i in range(mask_tensor.size()[0]):
        ind = torch.where(mask_tensor[i] == 0)[0]
        if ind.size()[0] == 0:
            result.append(target_tensor[i][1:-1])
        else:
            eos_ind = ind[0].item() - 1
            result.append(torch.concatenate((target_tensor[i][1:eos_ind], target_tensor[i][eos_ind + 1:]), axis=0))

    new_tensor = torch.stack(result, axis=0)
    return new_tensor


class Encoder(nn.Module):
    def __init__(self, configs, model_name='facebook/esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'esm_v2':
            self.model = prepare_esm_model(model_name, configs)
        # self.pooling_layer = nn.AdaptiveAvgPool2d((None, 1))
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.ParallelLinearDecoders = ParallelLinearDecoders(input_size=self.model.config.hidden_size,
                                                             output_sizes=[1] * configs.encoder.num_classes)
        self.type_head = nn.Linear(self.model.embeddings.position_embeddings.embedding_dim, configs.encoder.num_classes)
        self.overlap = configs.encoder.frag_overlap

        # SupCon
        self.apply_supcon = configs.supcon.apply
        if self.apply_supcon:
            self.projection_head = LayerNormNet(configs)
            self.n_pos = configs.supcon.n_pos
            self.n_neg = configs.supcon.n_neg
            self.batch_size = configs.train_settings.batch_size
            # new code
            self.skip_esm = configs.supcon.skip_esm
            self.protein_embeddings = torch.load('5283_esm2_t33_650M_UR50D.pt')
            # new code
        # mini tools for supcon
        curdir_path = os.getcwd()
        tokenizer = prepare_tokenizer(configs, curdir_path)
        self.tools = {
            'composition': configs.encoder.composition,
            'tokenizer': tokenizer,
            'max_len': configs.encoder.max_len,
            'train_device': configs.train_settings.device,
            'prm4prmpro': configs.encoder.prm4prmpro
        }
        # self.mhatt = nn.MultiheadAttention(embed_dim=320, num_heads=10, batch_first=True)
        # self.attheadlist = []
        # self.headlist = []
        # for i in range(9):
        # self.attheadlist.append(nn.MultiheadAttention(embed_dim=320, num_heads=1, batch_first=True))
        # self.headlist.append(nn.Linear(320, 1))
        # self.device = device
        # self.device=configs.train_settings.device

    def get_pro_emb(self, id, id_frags_list, seq_frag_tuple, emb_frags, overlap):
        # print(seq_frag_tuple)
        emb_pro_list = []
        for id_protein in id:
            ind_frag = 0
            id_frag = id_protein + "@" + str(ind_frag)
            while id_frag in id_frags_list:
                ind = id_frags_list.index(id_frag)
                emb_frag = emb_frags[ind]  # [maxlen-2, dim]
                seq_frag = seq_frag_tuple[ind]
                l = len(seq_frag)
                if ind_frag == 0:
                    emb_pro = emb_frag[:l]
                else:
                    overlap_emb = (emb_pro[-overlap:] + emb_frag[:overlap]) / 2
                    emb_pro = torch.concatenate((emb_pro[:-overlap], overlap_emb, emb_frag[overlap:l]), axis=0)
                ind_frag += 1
                id_frag = id_protein + "@" + str(ind_frag)
            # print('-before mean', emb_pro.shape)
            emb_pro = torch.mean(emb_pro, dim=0)
            # print('-after mean', emb_pro.shape)
            emb_pro_list.append(emb_pro)
        return emb_pro_list

    def forward(self, encoded_sequence, id, id_frags_list, seq_frag_tuple, pos_neg, warm_starting):
        """
        Batch is built before forward(), in train_loop()
        Batch is either (anchor) or (anchor+pos+neg)

        if apply supcon:
            if not warming starting:
                if pos_neg is None: ------------------> batch should be built as (anchor) in train_loop()
                    "CASE A"
                    get motif_logits from batch
                    get classification_head from batch
                else: --------------------------------> batch should be built as (anchor+pos+neg) in train_loop()
                    "CASE B"
                    get motif_logits from batch
                    get classification_head from batch
                    get projection_head from batch
            else: ------------------------------------> batch should be built as (anchor+pos+neg) in train_loop()
                "CASE C"
                get projection_head from batch
        else: ----------------------------------------> batch should be built as (anchor) in train_loop()
            "CASE D"
            get motif_logits from batch
            get classification_head from batch
        """
        classification_head = None
        motif_logits = None
        projection_head = None

        # features = self.model(input_ids=encoded_sequence['input_ids'],
        #                       attention_mask=encoded_sequence['attention_mask'])
        # last_hidden_state = remove_s_e_token(features.last_hidden_state,
        #                                      encoded_sequence['attention_mask'])  # [batch, maxlen-2, dim]
        # emb_pro_list = self.get_pro_emb(id, id_frags_list, seq_frag_tuple, last_hidden_state, self.overlap)
        # emb_pro = torch.stack(emb_pro_list, dim=0)  # [sample, dim]
        # new code
        if self.skip_esm:
            emb_pro_list = []
            for i in id:
                emb_pro_list.append(self.protein_embeddings[i])
            emb_pro = torch.stack(emb_pro_list, dim=0)
        else:
            features = self.model(input_ids=encoded_sequence['input_ids'],
                                  attention_mask=encoded_sequence['attention_mask'])
            last_hidden_state = remove_s_e_token(features.last_hidden_state,
                                                 encoded_sequence['attention_mask'])  # [batch, maxlen-2, dim]
            """"add something here? the 1+n_pos+n_neg"""
            emb_pro_list = self.get_pro_emb(id, id_frags_list, seq_frag_tuple, last_hidden_state, self.overlap)
            emb_pro = torch.stack(emb_pro_list, dim=0)  # [sample, dim]
        # new code

        if self.apply_supcon:
            if not warm_starting:
                motif_logits = self.ParallelLinearDecoders(last_hidden_state)
                motif_logits = torch.stack(motif_logits, dim=1).squeeze(-1)  # [batch, num_class, maxlen-2]
                classification_head = self.type_head(emb_pro)  # [sample, num_class]
                if pos_neg is not None:
                    """CASE B, when this if condition is skipped, CASE A"""
                    emb_pro_ = emb_pro.view((self.batch_size, 1 + self.n_pos + self.n_neg, -1))
                    projection_head = self.projection_head(emb_pro_)
            else:
                """CASE C"""
                n_batch = 8
                bch_anchors, bch_positives, bch_negatives = torch.split(emb_pro, [n_batch, n_batch*self.n_pos, n_batch*self.n_neg], dim=0)
                print(bch_anchors.shape)
                print(bch_positives.shape)
                print(bch_negatives.shape)
                print(emb_pro.shape)
                for i in range(n_batch):
                    anchor = bch_anchors[i].unsqueeze(0)
                    positive = bch_positives[(i * self.n_pos):(i * self.n_pos + self.n_pos)]
                    negative = bch_negatives[(i * self.n_neg):(i * self.n_neg + self.n_neg)]
                    print(anchor.shape)
                    print(positive.shape)
                    print(negative.shape)
                    triple = torch.cat((anchor, positive, negative), dim=0)
                    print(triple.shape)
                    exit(0)
                emb_pro_ = emb_pro.view((self.batch_size, 1 + self.n_pos + self.n_neg, -1))
                print(emb_pro_.shape)
                # projection_head = self.projection_head(emb_pro_)
                # exit(0)
                exit(0)
        else:
            """CASE D"""
            motif_logits = self.ParallelLinearDecoders(last_hidden_state)
            motif_logits = torch.stack(motif_logits, dim=1).squeeze(-1)  # [batch, num_class, maxlen-2]
            classification_head = self.type_head(emb_pro)  # [sample, num_class]
        return classification_head, motif_logits, projection_head


class Bothmodels(nn.Module):
    def __init__(self, configs, pretrain_loc, trainable_layers, model_name='facebook/esm2_t33_650M_UR50D',
                 model_type='esm_v2'):
        super().__init__()
        self.model_esm = prepare_esm_model(model_name, configs)
        self.model_promprot = initialize_PromptProtein(pretrain_loc, trainable_layers)
        self.pooling_layer = nn.AdaptiveAvgPool1d(output_size=1)
        self.head = nn.Linear(self.model_esm.embeddings.position_embeddings.embedding_dim + 1280,
                              configs.encoder.num_classes)
        self.cs_head = nn.Linear(self.model_esm.embeddings.position_embeddings.embedding_dim, 1)

    def forward(self, encoded_sequence):
        features = self.model_esm(input_ids=encoded_sequence["encoded_sequence_esm2"]['input_ids'],
                                  attention_mask=encoded_sequence["encoded_sequence_esm2"]['attention_mask'])
        transposed_feature = features.last_hidden_state.transpose(1, 2)
        pooled_features_esm2 = self.pooling_layer(transposed_feature).squeeze(2)

        cs_head = self.cs_head(features.last_hidden_state).squeeze(dim=-1)
        cs_pred = cs_head[:, 1:201]

        features = self.model_promprot(encoded_sequence["encoded_sequence_promprot"], with_prompt_num=1)['logits']
        transposed_feature = features.transpose(1, 2)
        pooled_features_promprot = self.pooling_layer(transposed_feature).squeeze(2)

        pooled_features = torch.cat((pooled_features_esm2, pooled_features_promprot), dim=1)

        classification_head = self.head(pooled_features)

        return classification_head, cs_pred


def prepare_models(configs, logfilepath, curdir_path):
    if configs.encoder.composition == "esm_v2":
        encoder = Encoder(model_name=configs.encoder.model_name,
                          model_type=configs.encoder.model_type,
                          configs=configs
                          )
    elif configs.encoder.composition == "promprot":
        encoder = CustomPromptModel(configs=configs,
                                    pretrain_loc=os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"),
                                    trainable_layers=["layers.32", "emb_layer_norm_after"])
    elif configs.encoder.composition == "both":
        encoder = Bothmodels(configs=configs,
                             pretrain_loc=os.path.join(curdir_path, "PromptProtein", "PromptProtein.pt"),
                             trainable_layers=[], model_name=configs.encoder.model_name,
                             model_type=configs.encoder.model_type)
    if not logfilepath == "":
        print_trainable_parameters(encoder, logfilepath)
    return encoder

# if __name__ == '__main__':
