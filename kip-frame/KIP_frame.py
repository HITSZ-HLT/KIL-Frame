import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from config import Config
from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.softmax_classifier import Softmax_Layer
# from model.memory_network.attention_memory import Attention_Memory
from model.memory_network.attention_memory_simplified import Attention_Memory_Simplified
from utils import outputer
from sampler import data_sampler, data_sampler_cluster_split
from data_loader import get_data_loader
import torch.nn.functional as F

def transfer_to_device(list_ins, device):
    import torch
    for ele in list_ins:
        if isinstance(ele, list):
            for x in ele:
                x.to(device)
        if isinstance(ele, torch.Tensor):
            ele.to(device)
    return list_ins


# Done
def get_sample_features(config, encoder, mem_set, feature_type="mask"):
    # aggregate the prototype set for further use.
    data_loader = get_data_loader(config, mem_set, False, False, 1)

    features = []
    for step, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
        with torch.no_grad():
            if feature_type == "mask":
                _,feature = encoder(input_ids=input_ids.to(config.device),
                                  token_type_ids=token_type_ids.to(config.device),
                                  attention_mask=attention_mask.to(config.device),
                                    return_mask_hidden=True)
            elif feature_type == "cls":
                feature = encoder.encoder.bert(input_ids=input_ids.to(config.device),
                                  token_type_ids=token_type_ids.to(config.device),
                                  attention_mask=attention_mask.to(config.device)).last_hidden_state[:, 0, :]
            else:
                assert  feature_type in ["mask","cls"]

            features.append(feature)

    features = torch.cat(features, dim=0)
    # proto = torch.mean(features, dim=0, keepdim=True)
    return features



def get_label_proto(config, rel_description, tokenizer, proto_patern='cls'):
    # aggregate the prototype set for further use.
    features = []
    encode_result = tokenizer.batch_encode_plus([rel_description], return_token_type_ids=True,
                                                max_length=64,
                                                truncation=True, padding='max_length',
                                                return_tensors='pt')
    with torch.no_grad():
        out = encoder.encoder.bert(input_ids=encode_result['input_ids'].to(config.device),
                                   token_type_ids=encode_result['token_type_ids'].to(config.device),
                                   attention_mask=encode_result['attention_mask'].to(config.device))
        if proto_patern == "cls":
            feature = out.last_hidden_state[:, 0, :]
        elif proto_patern == "avg":
            feature = torch.mean(out.last_hidden_state, dim=1)
        else:
            assert proto_patern in ["cls", "avg"]
        features.append(feature)
    features = torch.cat(features, dim=0)
    proto = torch.mean(features, dim=0, keepdim=True)
    # return the averaged prototype
    return proto



def get_att_proto(config, encoder, rel_description, tokenizer, mem_set, proto_patern='cls'):
    
    def multi_head(input,mask_last_one=False,num_head=12):
        input = input.view(1, -1, num_head, int(config.encoder_output_size/num_head))
        q_transpose = input.permute(0, 2, 1, 3)
        k_transpose = input.permute(0, 2, 1, 3)
        v_transpose = input.permute(0, 2, 1, 3)
        q_transpose *= (float(config.encoder_output_size/num_head) ** -0.5)
        # make it [B, H, N, N]
        dot_product = torch.matmul(q_transpose, k_transpose.permute(0, 1, 3, 2))
        if mask_last_one:
            dot_product[:,:,:,-1] = -1e7
        weights = F.softmax(dot_product, dim=-1)
        # output is [B, H, N, V]
        weighted_output = torch.matmul(weights, v_transpose)
        output_transpose = weighted_output.permute(0, 2, 1, 3).contiguous()
        output_transpose = output_transpose.view(-1, config.encoder_output_size)
        return output_transpose[-1:,]
    
    prompt_proto = get_label_proto(config, rel_description, tokenizer, proto_patern)
    sample_features = get_sample_features(config, encoder, mem_set,feature_type="cls")
    input = torch.cat([sample_features, prompt_proto], dim=0)
    
    last_hidden = multi_head(input=input, mask_last_one=config.proto_wo_prompt, num_head=config.num_head)
    
    if config.proto_with_ln:
        Ln=torch.nn.LayerNorm([1, config.encoder_output_size],elementwise_affine=False)
        return Ln(last_hidden)
    else:
        return last_hidden


# Use K-Means to select what samples to save, similar to at_least = 0
def select_data(config, encoder, sample_set, sampling_method='kmeans'):
    if len(sample_set) == 0:
        return []

    data_loader = get_data_loader(config, sample_set, shuffle=False, drop_last=False, batch_size=1)
    features = []

    for step, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
        with torch.no_grad():
            _, feature = encoder(input_ids=input_ids.to(config.device),
                                 token_type_ids=token_type_ids.to(config.device),
                                 attention_mask=attention_mask.to(config.device), return_mask_hidden=True)
        features.append(feature.cpu())

    features = np.concatenate(features)
    num_clusters = min(config.num_protos, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

    mem_set = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = sample_set[sel_index]
        mem_set.append(instance)
    return mem_set



def evaluate_strict_model(config, encoder, memory_network, test_data, protos4eval, seen_relations):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    memory_network.eval()
    seen_relation_ids = [rel2id[relation] for relation in seen_relations]
    with torch.no_grad():
        n = len(test_data)
        correct = 0
        protos4eval.unsqueeze(0)
        protos4eval = protos4eval.expand(data_loader.batch_size, -1, -1)
        for step, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):

            mem_for_batch = protos4eval.clone()
            labels = labels.to(config.device)
            logits_ori, mask_hidden = encoder(input_ids=input_ids.to(config.device),
                                              token_type_ids=token_type_ids.to(config.device),
                                              attention_mask=attention_mask.to(config.device), return_mask_hidden=True)
            ## add a memory_network
            reps = memory_network(mask_hidden, mem_for_batch)

            logits_mem = encoder.mlm_forward(reps)
            logits = logits_ori * (1 - config.mem_alpha) + logits_mem * config.mem_alpha

            seen_sim = logits[:, seen_relation_ids].cpu().data.numpy()
            max_smi = np.max(seen_sim, axis=1)
            label_smi = logits[:, labels].cpu().data.numpy()
            if label_smi >= max_smi:
                correct += 1
    return correct / n


def evaluate_loose_model(config, encoder, memory_network, test_data, protos4eval, seen_relations):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    memory_network.eval()
    seen_relation_ids = [rel2id[relation] for relation in seen_relations]
    with torch.no_grad():
        n = len(test_data)
        correct = 0
        protos4eval.unsqueeze(0)
        protos4eval = protos4eval.expand(data_loader.batch_size, -1, -1)
        for step, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):

            mem_for_batch = protos4eval.clone()
            labels = labels.to(config.device)
            logits_ori, mask_hidden = encoder(input_ids=input_ids.to(config.device),
                                              token_type_ids=token_type_ids.to(config.device),
                                              attention_mask=attention_mask.to(config.device), return_mask_hidden=True)
            ## add a memory_network
            reps = memory_network(mask_hidden, mem_for_batch)

            logits_mem = encoder.mlm_forward(reps)
            logits = logits_ori * (1 - config.mem_alpha) + logits_mem * config.mem_alpha

            random.shuffle(seen_relation_ids)
            random10_ids = seen_relation_ids[:10]
            seen_sim = logits[:, random10_ids].cpu().data.numpy()
            max_smi = np.max(seen_sim, axis=1)
            label_smi = logits[:, labels].cpu().data.numpy()
            if label_smi >= max_smi:
                correct += 1
    return correct / n


def train_simple_model(config, encoder, training_data, epochs):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': config.encoder_lr},
    ])

    for epoch_i in range(epochs):
        losses = []
        for step, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
            encoder.zero_grad()
            labels = labels.to(config.device)
            # [B, 40] N:candidates number
            logits = encoder(input_ids=input_ids.to(config.device), token_type_ids=token_type_ids.to(config.device),
                             attention_mask=attention_mask.to(config.device))

            loss = criterion(logits, labels)

            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)

            optimizer.step()
        print(f"Current task epoch-{epoch_i} loss is {np.array(losses).mean():.3}")


def get_debias_score(config, mem_data, cls_hidden, labels):
    """
    mem_data:[rels_num,768]
    cls_hidden:[B,768]
    """

    mem_data_full = torch.zeros([config.num_of_relation, config.encoder_output_size]).to(config.device)
    mem_data_full[protos_ids, :] = mem_data.clone()

    B = cls_hidden.shape[0]

    index_x = torch.LongTensor(list(range(B))).to(config.device)
    # [B,seen_rels_num]
    score = torch.matmul(cls_hidden, mem_data_full.t())
    score = torch.softmax(score, dim=-1)
    weight = torch.ones([B, config.num_of_relation], device=config.device)
    weight[:, current_rel_ids] = config.score_weight
    # [B,]
    score = weight[index_x, labels] * score[index_x, labels] + (1 - score[index_x, labels])
    return score


def train_mem_model(config, encoder, memory_network, training_data, mem_data, epochs):
    data_loader = get_data_loader(config=config, data=training_data, batch_size=config.replay_bs)
    encoder.train()

    memory_network.train()

    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': config.encoder_lr},
        {'params': memory_network.parameters(), 'lr': config.mem_lr}
    ])

    # mem_data.unsqueeze(0)
    # mem_data = mem_data.expand(data_loader.batch_size, -1, -1)
    for epoch_i in range(epochs):
        losses = []
        for step, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
            mem_for_batch = mem_data.clone()
            mem_for_batch.unsqueeze(0)
            mem_for_batch = mem_for_batch.expand(input_ids.shape[0], -1, -1)

            encoder.zero_grad()
            memory_network.zero_grad()

            labels = labels.to(config.device)

            logits_ori, mask_hidden, cls_hidden,loss_mlm = encoder.mask_replay_forward(input_ids=input_ids.to(config.device),
                                                          token_type_ids=token_type_ids.to(config.device),
                                                          attention_mask=attention_mask.to(config.device),
                                                          return_mask_hidden=True,
                                                          return_cls_hidden=True)
            ## add a memory_network
            reps = memory_network(mask_hidden, mem_for_batch)
            logits_mem = encoder.mlm_forward(reps)
            logits = logits_ori * (1 - config.mem_alpha) + logits_mem * config.mem_alpha
            score = get_debias_score(config, mem_data, cls_hidden, labels)
            loss_rel = torch.mean(criterion(logits, labels) * score)
            loss = loss_rel+loss_mlm*config.mlm_loss_weight
            losses.append(loss_rel.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)

            torch.nn.utils.clip_grad_norm_(memory_network.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f"Memory Replay epoch-{epoch_i} loss is {np.array(losses).mean():.3}")


if __name__ == '__main__':

    parser = ArgumentParser(
        description="Config for lifelong relation extraction (classification)")
    parser.add_argument('--config',
                        default='config_dir/debug.ini')

    args = parser.parse_args()
    config = Config(args.config)

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    # output result
    printer = outputer()
    middle_printer = outputer()
    start_printer = outputer()

    # set training batch
    for i in range(config.total_round):

        test_cur = []
        test_total = []

        # set random seed
        random.seed(config.seed + i * 100)

        # sampler setup
        if config.cluster_split_data:
            sampler = data_sampler_cluster_split(config=config, seed=config.seed + i * 100)
        else:
            sampler = data_sampler(config=config, seed=config.seed + i * 100)

        id2rel = sampler.id2rel
        rel2id = sampler.rel2id

        # encoder setup
        if hasattr(config, 'random_init_new_tokens'):
            if config.random_init_new_tokens:
                encoder = Bert_Encoder(rels_num=len(id2rel),
                                       device=config.device,
                                       chk_path=config.bert_path,
                                       id2name=None,
                                       tokenizer=sampler.tokenizer,
                                       init_by_cls=None,
                                       config=config)
            else:
                assert config.random_init_new_tokens, "if set random_init_new_tokens, must be true."
        else:
            encoder = Bert_Encoder(rels_num=len(id2rel),
                                   device=config.device,
                                   chk_path=config.bert_path,
                                   id2name=id2rel,
                                   tokenizer=sampler.tokenizer,
                                   init_by_cls=config.init_new_token_by_cls,
                                   config=config)

        # record testing results
        sequence_results = []
        result_whole_test = []

        # initialize memory and prototypes
        num_class = len(sampler.id2rel)
        memorized_samples = {}

        # load data and start computation
        for steps, (
                training_data, valid_data, test_data, current_relations, historic_test_data,
                seen_relations) in enumerate(
            sampler):

            print(current_relations)
            seen_ids = [id for id in range(len(id2rel)) if id2rel[id] in seen_relations]

            temp_mem = {}
            temp_protos = []
            protos_ids = []
            for relation in seen_relations:
                if relation not in current_relations:
                    temp_protos.append(get_att_proto(config, encoder, relation,
                                                     sampler.tokenizer,
                                                     memorized_samples[relation],
                                                     config.proto_patern))
                    protos_ids.append(rel2id[relation])

            # Initial
            train_data_for_initial = []
            for relation in current_relations:
                train_data_for_initial += training_data[relation]
            # train model

            train_simple_model(config, encoder, train_data_for_initial, config.step1_epochs)  # config.step1_epochs ->1

            for relation in current_relations:
                temp_mem[relation] = select_data(config, encoder, training_data[relation], config.sampling_method)
                temp_protos.append(get_att_proto(config, encoder, relation,
                                                 sampler.tokenizer,
                                                 temp_mem[relation],
                                                 config.proto_patern))
                protos_ids.append(rel2id[relation])
            temp_protos = torch.cat(temp_protos, dim=0).detach()



            memory_network = Attention_Memory_Simplified(mem_slots=len(seen_relations),
                                                         input_size=encoder.output_size,
                                                         output_size=encoder.output_size,
                                                         key_size=config.key_size,
                                                         head_size=config.head_size
                                                         ).to(config.device)

            # generate training data for the corresponding memory model (ungrouped)
            train_data_for_memory = []
            for relation in temp_mem.keys():
                train_data_for_memory += temp_mem[relation]
            for relation in memorized_samples.keys():
                train_data_for_memory += memorized_samples[relation]
            random.shuffle(train_data_for_memory)
            print(f"Memory training samples: {len(train_data_for_memory)}")

            current_rel_ids = [id for id in range(len(id2rel)) if id2rel[id] in current_relations]
            train_mem_model(config, encoder, memory_network, train_data_for_memory, temp_protos, config.step3_epochs)

            # regenerate memory
            for relation_index, relation in enumerate(current_relations):
                memorized_samples[relation] = select_data(config, encoder, training_data[relation],
                                                          config.sampling_method)

            protos4eval = []
            for relation in memorized_samples:
                protos4eval.append(get_att_proto(config, encoder, relation,
                                                     sampler.tokenizer,
                                                     memorized_samples[relation],
                                                     config.proto_patern))
            protos4eval = torch.cat(protos4eval, dim=0).detach()

            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]


            cur_acc = evaluate_strict_model(config, encoder, memory_network, test_data_1, protos4eval,
                                            seen_relations)
            total_acc = evaluate_strict_model(config, encoder, memory_network, test_data_2, protos4eval,
                                              seen_relations)


            print(f'Restart Num {i + 1}')
            print(f'task--{steps + 1}:')
            print(f'current test acc:{cur_acc}')
            print(f'history test acc:{total_acc}')
            test_cur.append(cur_acc)
            test_total.append(total_acc)
            print('current test acc', test_cur)
            print('history test acc', test_total)

