import torch
import regex
import random
import time
import quadprog
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn_extra.cluster import KMedoids
from dataset import Batch
from collections import Counter


def RANDOM(examples, memory_size):
    index_arr = np.arange(len(examples))
    np.random.shuffle(index_arr)

    ids = index_arr[:memory_size]
    selected_examples = [examples[i] for i in ids]
    return selected_examples

def FSS(model, examples, memory_size, args):
    model.eval()
    feature_vectors = torch.zeros(len(examples), args.encoder_dim)

    with torch.no_grad():
        perm = [x for x in range(len(examples))]
        st = 0
        while st < len(examples):
            ed = st + args.batch_size if st + args.batch_size < len(perm) else len(perm)
            batch = Batch(examples[st: ed], model.grammar, model.vocab, cuda=args.cuda, table_vocab=model.table_vocab)

            src_encodings, table_encoding, src_mask = model.encoding_src_col(batch, batch.src_sents_word,
                                                                             batch.table_sents_word,
                                                                             model.col_enc_n)
            src_pooling = max_pooling_by_mask(src_encodings, src_mask)
            feature_vectors[st:ed, :].copy_(src_pooling.detach())
            st = ed
    assert len(examples) == ed

    X = feature_vectors.cpu().numpy()

    km = KMeans(n_clusters=memory_size).fit(X)

    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)

    selected_examples = []
    for top_idx in closest.tolist():
        selected_examples.append(examples[top_idx])

    assert len(selected_examples) == memory_size
    model.train()
    return selected_examples

def GSS(model, examples, memory_data, observed_task_ids, args, task_id):
    model.train()
    for _task_id in range(len(observed_task_ids) - 1):
        model.zero_grad()
        past_task_id = observed_task_ids[_task_id]

        replay_examples = memory_data[past_task_id]

        report_loss, example_num, loss = self.train_one_batch(examples[st:ed], report_loss, example_num)

        replay_loss = epoch_train(model=agent.net,
                                  optimizer=None,
                                  batch_size=agent.args.batch_size,
                                  sql_data=replay_sql_data,
                                  table_data=replay_table_data,
                                  args=agent.args,
                                  epoch=0,
                                  loss_epoch_threshold=agent.args.loss_epoch_threshold,
                                  sketch_loss_coefficient=agent.args.sketch_loss_coefficient,
                                  loss_backward=False)
        replay_loss.backward()
        store_grad(agent.net.parameters,
                   agent.grads,
                   agent.grad_dims,
                   past_task_id)

    index_list = []
    perm = [i for i in range(len(train_sql_data))]
    for i in range(len(train_sql_data)):
        ptloss, _ = batch_train(model=agent.net,
                                optimizer=None,
                                perm=perm,
                                st=i,
                                ed=i + 1,
                                sql_data=train_sql_data,
                                table_data=train_table_data,
                                args=agent.args,
                                epoch=0,
                                loss_epoch_threshold=agent.args.loss_epoch_threshold,
                                sketch_loss_coefficient=agent.args.sketch_loss_coefficient,
                                loss_backward=False)
        ptloss.backward()

        store_grad(agent.net.parameters,
                   agent.grads,
                   agent.grad_dims,
                   task_id)
        indx = agent.net.new_long_tensor(agent.observed_tasks[:-1])

        dotp = torch.mm(agent.grads[:, task_id].unsqueeze(0), agent.grads.index_select(1, indx))
        if (dotp < 0).sum() == 0:  # or (dotp > 0.01).sum() == 0:
            index_list.append(i)
    return [train_sql_data[i] for i in index_list]

def PRIOR(model, examples, memory_size, args):
    model.eval()

    exemplar_cands = []
    with torch.no_grad():
        st = 0
        while st < len(examples):
            # training on the batch of current task
            ed = st + args.batch_size if st + args.batch_size < len(examples) else len(examples)

            score = model.forward(examples[st: ed])
            loss_sketch = -score[0]
            loss_lf = -score[1]

            _loss = loss_lf + loss_sketch

            _loss = _loss.tolist()

            assert ed - st == len(_loss)
            for i, data in enumerate(examples[st: ed]):
                exemplar_cands.append([data, _loss[i]])
            st = ed

    sorted(exemplar_cands, key=lambda x: x[-1], reverse=False)

    number = min(memory_size, len(exemplar_cands))

    selected_examples = []
    for i in range(number):
        selected_examples.append(exemplar_cands[i][0])
    model.train()
    return selected_examples

def BALANCE(examples, memory_size):
    col_count, col_example_dict = count_cols(examples)

    rebalance_examples = []

    while len(rebalance_examples) < memory_size:
        # print (count)
        col_index = random.randint(0, len(col_count.most_common()) - 1)
        col, count = col_count.most_common()[col_index]
        # print(template)
        col_examples = col_example_dict[col]
        if len(col_examples) > 0:
            # if template_type=='specific':
            index = random.randint(0, len(col_examples) - 1)
            rebalance_examples.append(col_examples.pop(index))

    return rebalance_examples

def LFS(examples, memory_size):
    sim_array = np.ndarray((len(examples), len(examples)))

    col_set_list = []
    for example in examples:
        col_set = get_used_cols(example)
        col_set_list.append(col_set)

    for out_idx, out_col_set in enumerate(col_set_list):
        for in_idx, in_col_set in enumerate(col_set_list):
            sim = cal_col_similarity(in_col_set, out_col_set)
            sim_array[out_idx, in_idx] = sim
            sim_array[in_idx, out_idx] = sim

    kmedoids = KMedoids(metric='precomputed', n_clusters=memory_size, init='k-medoids++').fit((2 - sim_array))
    # print (kmedoids.medoid_indices_)
    added_inds = kmedoids.medoid_indices_

    added_inds_list = added_inds.squeeze().tolist()

    selected_examples = [examples[indx] for indx in added_inds_list]

    return selected_examples

def DLFS(examples, memory_size):
    sim_array = np.ndarray((len(examples), len(examples)))

    col_set_list = []
    for example in examples:
        col_set = get_used_cols(example)
        col_set_list.append(col_set)

    for out_idx, out_col_set in enumerate(col_set_list):
        for in_idx, in_col_set in enumerate(col_set_list):
            sim = cal_col_similarity(in_col_set, out_col_set)
            sim_array[out_idx, in_idx] = sim
            sim_array[in_idx, out_idx] = sim

    sc = SpectralClustering(memory_size, affinity='precomputed')
    sc.fit(sim_array)

    labels = sc.labels_.tolist()
    # print (labels)
    index_list = [None] * memory_size

    for idx, example in enumerate(examples):
        label = labels[idx]
        index_list[label] = idx

    cols = []
    for col_set in col_set_list:
        for col in col_set:
            cols.append(col)

    col_to_id = {token: idx for idx, token in enumerate(set(cols))}

    freq = torch.zeros(len(examples), len(col_to_id))
    for idx, col_set in enumerate(col_set_list):
        for col in col_set:
            if col in col_to_id:
                freq[idx][col_to_id[col]] = 1

    freq_m = freq[index_list].clone().detach()
    freq_sum = freq_m.sum(dim=0)
    freq_prob = freq_sum / freq_m.sum()
    current_entropy = Categorical(probs=freq_prob).entropy()

    added_inds_list = index_list
    entropy_add = 1000
    while entropy_add > 0:

        entropy_add = 0
        for label_index in range(memory_size):
            entropy_tensor = []
            entropy_tensor_indx = []
            freq_list = []
            for train_idx in range(len(examples)):
                if labels[train_idx] == label_index:
                    freq_sum = freq_m.sum(dim=0)
                    freq_prob = freq_sum / freq_m.sum()
                    current_entropy = Categorical(probs=freq_prob).entropy()
                    example_freq = freq[train_idx]

                    # entropy_tensor = torch.zeros(n_memories)

                    temp_freq_sum = freq_sum - freq_m[label_index] + example_freq
                    temp_freq_prob = temp_freq_sum / temp_freq_sum.sum()
                    temp_entropy = Categorical(probs=temp_freq_prob).entropy()
                    entropy_tensor.append(temp_entropy - current_entropy)
                    entropy_tensor_indx.append(train_idx)
                    freq_list.append(example_freq)

            entropy_tensor = torch.Tensor(entropy_tensor)
            max_entropy, max_entropy_ind = torch.max(entropy_tensor, dim=-1)
            # print (max_entropy_ind.item())
            max_selected_indx = entropy_tensor_indx[max_entropy_ind.item()]
            if entropy_tensor[max_entropy_ind.item()].item() > 0 and not (max_selected_indx in added_inds_list):
                added_inds_list[label_index] = max_selected_indx
                freq_m[label_index] = freq_list[max_entropy_ind.item()]
                entropy_add += entropy_tensor[max_entropy_ind.item()].item()

        # print(entropy_add)

    # print(added_inds_list)
    freq_m = freq[added_inds_list].clone().detach()
    # print(freq_m.sum(dim=0))
    # print(freq_m.sum(dim=0).size())
    freq_sum = freq_m.sum(dim=0)
    freq_prob = freq_sum / freq_m.sum()
    current_entropy = Categorical(probs=freq_prob).entropy()
    # print(current_entropy)
    selected_examples = [examples[indx] for indx in added_inds_list]

    return selected_examples

def get_used_cols(example):
    # print(example.tgt_actions)
    rule_labels = " ".join([str(x) for x in example.tgt_actions])
    # print(rule_labels)
    pattern = regex.compile('C\(.*?\)')
    result_pattern = set(pattern.findall(rule_labels))
    used_cols = []
    for c in result_pattern:
        index = int(c[2:-1])
        col = " ".join([x for x in example.tab_cols[index]])
        used_cols.append(col)
    return used_cols

def cal_col_similarity(col_set1, col_set2):
    overlap = set(col_set1) & set(col_set2)
    score = len(overlap) / len(col_set1) + len(overlap) / len(col_set2)
    return score

def count_cols(examples):
    used_cols_list = []
    col_example_dict = {}
    for example in examples:
        used_cols = get_used_cols(example)
        # print(used_cols)
        used_cols_list.extend(used_cols)

        for col in used_cols:
            if col not in col_example_dict:
                col_example_dict[col] = [example]
            else:
                col_example_dict[col].append(example)

    col_count = Counter([col for col in used_cols_list])
    return col_count, col_example_dict

def store_grad(para, grads, grad_dims, task_id):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, task_id].fill_(0.0)
    cnt = 0
    for param in para():
        if param.grad is not None:
            st = 0 if cnt == 0 else sum(grad_dims[:cnt])
            ed = sum(grad_dims[:cnt + 1])
            grads[st: ed, task_id].copy_(param.grad.data.view(-1))
        cnt += 1

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def max_pooling_by_mask(seq, mask):
    mask = mask.unsqueeze(-1).expand_as(seq)
    seq = seq.masked_fill(mask.bool(), -1e18)
    return seq.max(dim=1)[0]
