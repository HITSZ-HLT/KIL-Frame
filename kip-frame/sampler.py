import numpy as np
import json
import random
from transformers import BertTokenizer
from tqdm import tqdm
class data_sampler(object):

    def __init__(self, config=None, seed=None):

        self.config = config
        
        # template for choose
        self.templates = ["In the following sentence, the relationship between ##entity_1## and ##entity_2## is [MASK].",
                          "The relationship between ##entity_1## and ##entity_2## is [MASK].",
                          "I think ##entity_1## is [MASK] of ##entity_2##.",
                          "In the following sentence, ##entity_1## is [MASK] of ##entity_2##.",
                          "##entity_1##? [MASK], ##entity_2##",
                          "I think [E11] is [MASK] of [E21].",
                          "I think [E11] ##entity_1## [E12] is [MASK] of [E21] ##entity_2## [E12].",
                          "[P00] The relationship between ##entity_1## and ##entity_2## is [MASK].",
                          "[P00] [E11] [E12] [E21] [E22] ##entity_1##? [MASK], ##entity_2##.",
                         ]
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, additional_special_tokens=["[P00]","[E11]", "[E12]", "[E21]", "[E22]"])

        # read relation data
        if hasattr(self.config,'use_label_description') and self.config.use_label_description:
            self.id2rel, self.rel2id,self.id2des,self.des2id= self._read_relations_pair(config.relation_file)
        else:
            self.id2rel, self.rel2id = self._read_relations(config.relation_file)

        # random sampling
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)
        if hasattr(self.config, 'few_shot'):
            self.training_dataset = [rel_data[:self.config.few_shot] for rel_data in self.training_dataset]
            print(f"Each relation data set upto {self.config.few_shot}")

        # generate the task number
        self.batch = 0
        self.task_length = len(self.id2rel) // self.config.rel_per_task

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

        

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            raise StopIteration()

        indexs = self.shuffle_index[self.config.rel_per_task*self.batch: self.config.rel_per_task*(self.batch+1)]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def _read_data(self, file):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_dataset = [[] for _ in range(self.config.num_of_relation)]
        val_dataset = [[] for _ in range(self.config.num_of_relation)]
        test_dataset = [[] for _ in range(self.config.num_of_relation)]
        for relation in tqdm(data.keys(),"Prepare data:"):
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):

                tokenized_sample = {}
                tokenized_sample['relation'] = self.rel2id[sample['relation']]

                p11=sample['tokens'].index("[E11]")+1
                p12 = sample['tokens'].index("[E12]")
                p21 = sample['tokens'].index("[E21]")+1
                p22 = sample['tokens'].index("[E22]")

                entity_1 = " ".join(sample['tokens'][p11:p12])
                entity_2 = " ".join(sample['tokens'][p21:p22])

                all_candidates = []
                
                if hasattr(self.config,'use_marker'):
                    if not self.config.use_marker:
                        sample_sentence = ' '.join(sample['tokens']).replace(" [E11] ", " ").replace(" [E12] ", " ").replace(
                            " [E21] ", " ").replace(" [E22] ", " ")
                else:
                    sample_sentence = ' '.join(sample['tokens'])

                template = self.templates[self.config.template_id].replace("##entity_1##",entity_1).replace("##entity_2##",entity_2)
                
                all_candidates.append((template,sample_sentence))

                ## return tensor
                tokenized_sample['tokens'] = self.tokenizer.batch_encode_plus(all_candidates, return_token_type_ids=True,
                                                                 max_length=self.config.max_length,
                                                                 truncation=True, padding='max_length',
                                                                 return_tensors='pt')

                if self.config.task_name == 'FewRel':
                    
                    if i < self.config.num_of_train:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < self.config.num_of_train + self.config.num_of_val:
                        val_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                else:
                    if i < len(rel_samples) // 5 and count <= 40:
                        count += 1
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)  
                        if count1 >= 320:
                            break         

        return train_dataset, val_dataset, test_dataset

    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id
    
    def _read_relations_pair(self, file):
        id2pairs = json.load(open(file, 'r', encoding='utf-8'))
        id2rel = []
        id2des = []
        rel2id = {}
        des2id = {}
        for id, item in enumerate(id2pairs):
            rel, des = item[0], item[1]
            id2rel.append(rel)
            rel2id[rel] = id
            id2des.append(des)
            des2id[des] = id
        return id2rel, rel2id, id2des, des2id

    
    
class data_sampler_cluster_split(object):

    def __init__(self, config=None, seed=None):

        self.config = config

        # template for choose
        self.templates = [
            "In the following sentence, the relationship between ##entity_1## and ##entity_2## is [MASK].",
            "The relationship between ##entity_1## and ##entity_2## is [MASK].",
            "I think ##entity_1## is [MASK] of ##entity_2##.",
            "In the following sentence, ##entity_1## is [MASK] of ##entity_2##.",
            "##entity_1##? [MASK], ##entity_2##",
            "In the following sentence, the implied relationship is [MASK]."
            ]

        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)

        # read relation data
        if hasattr(self.config, 'use_label_description') and self.config.use_label_description:
            self.id2rel, self.rel2id, self.id2des, self.des2id = self._read_relations_pair(config.relation_file)
        else:
            self.id2rel, self.rel2id = self._read_relations(config.relation_file)

        # random sampling
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        #         self.shuffle_index = list(range(len(self.id2rel)))
        #         random.shuffle(self.shuffle_index)
        #         self.shuffle_index = np.argsort(self.shuffle_index)
        if self.config.task_name == 'FewRel':
            self.shuffle_index = [[2, 4, 8, 11, 15, 21, 36, 37, 43, 57, 61, 63, 64, 75],
                                  [19, 33, 74, 76, 78],
                                  [1, 13, 14, 22, 24, 32, 38, 47, 58, 69, 73],
                                  [9, 34, 41, 42, 44, 72],
                                  [0, 5, 6, 7, 10, 12, 16, 26, 28, 29, 30, 31, 35, 45, 46, 50, 59, 60, 62, 65, 67, 71,
                                   77, 79],
                                  [17, 20, 27, 39, 54],
                                  [66],
                                  [48],
                                  [51, 55],
                                  [3, 18, 23, 25, 40, 49, 52, 53, 56, 68, 70]]
            self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)
        elif self.config.task_name == 'TACRED':
            # del idnex 17  from [2, 25, 28, 7, 13, 17, 20, 29, 38, 39],
            # because the auther del this data from the dataset
            self.shuffle_index = [[15],
                                  [4, 11, 18, 34, 36],
                                  [6, 8, 10, 12, 16, 37],
                                  [2, 25, 28, 7, 13, 20, 29, 38, 39],
                                  [0, 9, 14, 21, 31, 40],
                                  [22, 33],
                                  [1, 27],
                                  [5],
                                  [23, 26],
                                  [19, 24, 3, 30, 32, 35], ]
            self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)
            
        elif self.config.task_name == 'simpleqa':
            with open(self.config.index_file) as f:
                shuffle_index = json.load(f)
                self.shuffle_index=eval(shuffle_index)
                
            self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data_simpleqa(self.config.data_file)
        else:
            assert self.config.task_name in ['FewRel', 'TACRED','simpleqa'], f"{self.config.task_name} not in ['FewRel','TACRED','simpleqa']"

        # regenerate data
        

        # generate the task number
        self.batch = 0
        #         self.task_length = len(self.id2rel) // self.config.rel_per_task
        if self.config.task_name == 'FewRel':
            self.task_length = 10
        elif self.config.task_name == 'TACRED':
            self.task_length = 10
        elif self.config.task_name == 'simpleqa':
            self.task_length = 20

        # record relations
        self.seen_relations = []
        self.history_test_data = []

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            raise StopIteration()

        #         indexs = self.shuffle_index[self.config.rel_per_task*self.batch: self.config.rel_per_task*(self.batch+1)]
        indexs = self.shuffle_index[self.batch]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}
        self.history_test_data.append({})

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[-1][self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def _read_data(self, file):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_dataset = [[] for _ in range(self.config.num_of_relation)]
        val_dataset = [[] for _ in range(self.config.num_of_relation)]
        test_dataset = [[] for _ in range(self.config.num_of_relation)]
        for relation in tqdm(data.keys(), "Prepare data:"):
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):

                tokenized_sample = {}
                tokenized_sample['relation'] = self.rel2id[sample['relation']]

                p11 = sample['tokens'].index("[E11]") + 1
                p12 = sample['tokens'].index("[E12]")
                p21 = sample['tokens'].index("[E21]") + 1
                p22 = sample['tokens'].index("[E22]")

                entity_1 = " ".join(sample['tokens'][p11:p12])
                entity_2 = " ".join(sample['tokens'][p21:p22])

                all_candidates = []
                sample_sentence = ' '.join(sample['tokens']).replace(" [E11] ", " ").replace(" [E12] ", " ").replace(
                    " [E21] ", " ").replace(" [E22] ", " ")

                #                 template = f"In the following sentence, the relationship between {entity_1} and {entity_2} is [MASK]."
                template = self.templates[self.config.template_id].replace("##entity_1##", entity_1).replace(
                    "##entity_2##", entity_2)

                all_candidates.append((template, sample_sentence))

                ## return tensor
                tokenized_sample['tokens'] = self.tokenizer.batch_encode_plus(all_candidates,
                                                                              return_token_type_ids=True,
                                                                              max_length=self.config.max_length,
                                                                              truncation=True, padding='max_length',
                                                                              return_tensors='pt')

                if self.config.task_name == 'FewRel':

                    if i < self.config.num_of_train:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < self.config.num_of_train + self.config.num_of_val:
                        val_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                else:
                    if i < len(rel_samples) // 5 and count <= 40:
                        count += 1
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        if count1 >= 320:
                            break

        return train_dataset, val_dataset, test_dataset
    
    def _read_data_simpleqa(self, file):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_dataset = [[] for _ in range(self.config.num_of_relation)]
        val_dataset = [[] for _ in range(self.config.num_of_relation)]
        test_dataset = [[] for _ in range(self.config.num_of_relation)]
        for relation in tqdm(data.keys(), "Prepare data:"):
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):

                tokenized_sample = {}
                tokenized_sample['relation'] = self.rel2id[sample['relation']]


                all_candidates = []
                sample_sentence = ' '.join(sample['tokens'])
                template = self.templates[self.config.template_id]
                all_candidates.append((template, sample_sentence))
                ## return tensor
                tokenized_sample['tokens'] = self.tokenizer.batch_encode_plus(all_candidates,
                                                                              return_token_type_ids=True,
                                                                              max_length=self.config.max_length,
                                                                              truncation=True, padding='max_length',
                                                                              return_tensors='pt')

                if i < len(rel_samples)*0.2:
                    count += 1
                    test_dataset[self.rel2id[relation]].append(tokenized_sample)
                else:
                    count1 += 1
                    train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    if count1 >= len(rel_samples)*0.7:
                        break

        return train_dataset, val_dataset, test_dataset
    
    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id

    def _read_relations_pair(self, file):
        id2pairs = json.load(open(file, 'r', encoding='utf-8'))
        id2rel = []
        id2des = []
        rel2id = {}
        des2id = {}
        for id, item in enumerate(id2pairs):
            rel, des = item[0], item[1]
            id2rel.append(rel)
            rel2id[rel] = id
            id2des.append(des)
            des2id[des] = id
        return id2rel, rel2id, id2des, des2id
    


class data_sampler_raw(object):

    def __init__(self, config=None, seed=None):

        self.config = config

        # template for choose
        self.templates = [
            "In the following sentence, the relationship between ##entity_1## and ##entity_2## is [MASK].",
            "The relationship between ##entity_1## and ##entity_2## is [MASK].",
            "I think ##entity_1## is [MASK] of ##entity_2##.",
            "In the following sentence, ##entity_1## is [MASK] of ##entity_2##.",
            ]

        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)

        # read relation data
        if hasattr(self.config, 'use_label_description') and self.config.use_label_description:
            self.id2rel, self.rel2id, self.id2des, self.des2id = self._read_relations_pair(config.relation_file)
        else:
            self.id2rel, self.rel2id = self._read_relations(config.relation_file)

        # random sampling
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)

        # generate the task number
        self.batch = 0
        self.task_length = len(self.id2rel) // self.config.rel_per_task

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            raise StopIteration()

        indexs = self.shuffle_index[self.config.rel_per_task * self.batch: self.config.rel_per_task * (self.batch + 1)]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def _read_data(self, file):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_dataset = [[] for _ in range(self.config.num_of_relation)]
        val_dataset = [[] for _ in range(self.config.num_of_relation)]
        test_dataset = [[] for _ in range(self.config.num_of_relation)]
        for relation in tqdm(data.keys(), "Prepare data:"):
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):

                tokenized_sample = {}
                tokenized_sample['relation'] = self.rel2id[sample['relation']]

                p11 = sample['tokens'].index("[E11]") + 1
                p12 = sample['tokens'].index("[E12]")
                p21 = sample['tokens'].index("[E21]") + 1
                p22 = sample['tokens'].index("[E22]")

                entity_1 = " ".join(sample['tokens'][p11:p12])
                entity_2 = " ".join(sample['tokens'][p21:p22])


                sample_sentence = ' '.join(sample['tokens']).replace("[E11] ", "").replace(" [E12]", "").replace(
                    "[E21] ", "").replace(" [E22]", "")

                #                 template = f"In the following sentence, the relationship between {entity_1} and {entity_2} is [MASK]."
                template = self.templates[self.config.template_id].replace("##entity_1##", entity_1).replace(
                    "##entity_2##", entity_2)

                ## return tensor
                tokenized_sample['tokens'] = (template, sample_sentence)
                # tokenized_sample['tokens'] = self.tokenizer.batch_encode_plus(all_candidates,
                #                                                               return_token_type_ids=True,
                #                                                               max_length=self.config.max_length,
                #                                                               truncation=True, padding='max_length',
                #                                                               return_tensors='pt')

                if self.config.task_name == 'FewRel':
                    if i < self.config.num_of_train:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < self.config.num_of_train + self.config.num_of_val:
                        val_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                else:
                    if i < len(rel_samples) // 5 and count <= 40:
                        count += 1
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        if count1 >= 320:
                            break

        return train_dataset, val_dataset, test_dataset

    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id

    def _read_relations_pair(self, file):
        id2pairs = json.load(open(file, 'r', encoding='utf-8'))
        id2rel = []
        id2des = []
        rel2id = {}
        des2id = {}
        for id, item in enumerate(id2pairs):
            rel, des = item[0], item[1]
            id2rel.append(rel)
            rel2id[rel] = id
            id2des.append(des)
            des2id[des] = id
        return id2rel, rel2id, id2des, des2id


class data_sampler_order(object):

    def __init__(self, config=None,order=0 ,seed=None):

        self.config = config
        self.order = order
        # template for choose
        self.templates = [
            "In the following sentence, the relationship between ##entity_1## and ##entity_2## is [MASK].",
            "The relationship between ##entity_1## and ##entity_2## is [MASK].",
            "I think ##entity_1## is [MASK] of ##entity_2##.",
            "In the following sentence, ##entity_1## is [MASK] of ##entity_2##.",
            "##entity_1##? [MASK], ##entity_2##",
            "I think [E11] is [MASK] of [E21].",
            "I think [E11] ##entity_1## [E12] is [MASK] of [E21] ##entity_2## [E12].",
            "[P00] The relationship between ##entity_1## and ##entity_2## is [MASK].",
            "[P00] [E11] [E12] [E21] [E22] ##entity_1##? [MASK], ##entity_2##.",
            ]
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path,
                                                       additional_special_tokens=["[P00]", "[E11]", "[E12]", "[E21]",
                                                                                  "[E22]"])

        # read relation data
        if hasattr(self.config, 'use_label_description') and self.config.use_label_description:
            self.id2rel, self.rel2id, self.id2des, self.des2id = self._read_relations_pair(config.relation_file)
        else:
            self.id2rel, self.rel2id = self._read_relations(config.relation_file)

        # random sampling
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)
        if hasattr(self.config, 'few_shot'):
            self.training_dataset = [rel_data[:self.config.few_shot] for rel_data in self.training_dataset]
            print(f"Each relation data set upto {self.config.few_shot}")

        # generate the task number
        self.batch = 0
        self.task_length = len(self.id2rel) // self.config.rel_per_task

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.task_length:
            raise StopIteration()

        #[order, order+1, ... , ...]
        batch_order = (self.batch + self.order) % self.task_length
        indexs = self.shuffle_index[self.config.rel_per_task * batch_order: self.config.rel_per_task * (batch_order + 1)]

        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def _read_data(self, file):
        '''
        :param file: the input sample file
        :return: samples for the model: [relation label, text]
        '''
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_dataset = [[] for _ in range(self.config.num_of_relation)]
        val_dataset = [[] for _ in range(self.config.num_of_relation)]
        test_dataset = [[] for _ in range(self.config.num_of_relation)]
        for relation in tqdm(data.keys(), "Prepare data:"):
            rel_samples = data[relation]
            if self.seed != None:
                random.seed(self.seed)
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):

                tokenized_sample = {}
                tokenized_sample['relation'] = self.rel2id[sample['relation']]

                p11 = sample['tokens'].index("[E11]") + 1
                p12 = sample['tokens'].index("[E12]")
                p21 = sample['tokens'].index("[E21]") + 1
                p22 = sample['tokens'].index("[E22]")

                entity_1 = " ".join(sample['tokens'][p11:p12])
                entity_2 = " ".join(sample['tokens'][p21:p22])

                all_candidates = []

                if hasattr(self.config, 'use_marker'):
                    if not self.config.use_marker:
                        sample_sentence = ' '.join(sample['tokens']).replace(" [E11] ", " ").replace(" [E12] ",
                                                                                                     " ").replace(
                            " [E21] ", " ").replace(" [E22] ", " ")
                else:
                    sample_sentence = ' '.join(sample['tokens'])

                template = self.templates[self.config.template_id].replace("##entity_1##", entity_1).replace(
                    "##entity_2##", entity_2)

                all_candidates.append((template, sample_sentence))

                ## return tensor
                tokenized_sample['tokens'] = self.tokenizer.batch_encode_plus(all_candidates,
                                                                              return_token_type_ids=True,
                                                                              max_length=self.config.max_length,
                                                                              truncation=True, padding='max_length',
                                                                              return_tensors='pt')

                if self.config.task_name == 'FewRel':

                    if i < self.config.num_of_train:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < self.config.num_of_train + self.config.num_of_val:
                        val_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                else:
                    if i < len(rel_samples) // 5 and count <= 40:
                        count += 1
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        if count1 >= 320:
                            break

        return train_dataset, val_dataset, test_dataset

    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id

    def _read_relations_pair(self, file):
        id2pairs = json.load(open(file, 'r', encoding='utf-8'))
        id2rel = []
        id2des = []
        rel2id = {}
        des2id = {}
        for id, item in enumerate(id2pairs):
            rel, des = item[0], item[1]
            id2rel.append(rel)
            rel2id[rel] = id
            id2des.append(des)
            des2id[des] = id
        return id2rel, rel2id, id2des, des2id