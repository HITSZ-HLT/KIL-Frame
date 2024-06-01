from transformers import BertForMaskedLM
import torch.nn as nn
import torch

class Bert_Encoder(nn.Module):

    def __init__(self, rels_num=0, device='cpu', chk_path=None,id2name=None,tokenizer=None,init_by_cls=False,config=None):
        super(Bert_Encoder, self).__init__()

        self.rels_num = rels_num
        self.device = device
        self.encoder = BertForMaskedLM.from_pretrained(chk_path)
        self.config = config
        ## 1 is remain for replay-token, 4 is remain for entity marker
        self.encoder.resize_token_embeddings(self.encoder.config.vocab_size + 1+4 +rels_num)
        self.output_size = config.encoder_output_size
        self.tokenizer = tokenizer
        self.replay_token_id = torch.LongTensor([self.encoder.config.vocab_size-(5+rels_num)]).to(device)

        if config.p_mask>=0:
            self.p_mask = config.p_mask
            self.mlm_loss_fn = nn.CrossEntropyLoss()
        if id2name: ## init the new token embedding
            candidate_labels = id2name

            if init_by_cls:
#                 encode_result = tokenizer.batch_encode_plus(candidate_labels, return_token_type_ids=True,
#                                                                  max_length=64,
#                                                                  truncation=True, padding='max_length',
#                                                                  return_tensors='pt')

#                 out=self.encoder.bert(input_ids=encode_result['input_ids'],
#                                       attention_mask=encode_result['attention_mask'],token_type_ids=
#                                       encode_result['token_type_ids'])
#                 init_values = out.last_hidden_state[:,0,:]
#                 std1 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
#                 self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]=init_values
#                 std2 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
#                 print(f"Init the new tokens embedding by [CLS] token of labels description. The variance from {std1:.4} to {std2:.4} ")

                ## attention avg of embedding
                candidate_ids = tokenizer.batch_encode_plus(candidate_labels, add_special_tokens=False)['input_ids']
                std1 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                for i in range(rels_num):
                    label_token_ids = candidate_ids[i]
                    ## [1,12,L,L]
                    attentions=self.encoder.bert(input_ids=candidate_ids,output_attentions=True).attentions
                    
                    label_emb_init  = self.encoder.bert.embeddings.word_embeddings.weight.data[label_token_ids].mean(dim=0)
                    
                    self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:][i] = label_emb_init
                std2 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                print(f"Init the new tokens embedding by attention-weighted-avarage of labels token-emb. The variance from {std1:.4} to {std2:.4} ")
                
            else:
                candidate_ids = tokenizer.batch_encode_plus(candidate_labels, add_special_tokens=False)['input_ids']
                std1 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                for i in range(rels_num):
                    label_token_ids = candidate_ids[i]
                    label_emb_init  = self.encoder.bert.embeddings.word_embeddings.weight.data[label_token_ids].mean(dim=0)
                    
                    self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:][i] = label_emb_init
                std2 = torch.std(self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]).cpu().item()
                self.label_emb_init = self.encoder.bert.embeddings.word_embeddings.weight.data[-rels_num:]
                print(f"Init the new tokens embedding by avarage of labels token-emb. The variance from {std1:.4} to {std2:.4} ")
        else:
            print(f"Random init the new tokens embedding.")
            
        self.to(self.device)

    def forward(self, input_ids,attention_mask,token_type_ids,return_mask_hidden=False,return_cls_hidden=False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids,output_hidden_states=return_mask_hidden )

        ## [MASK] position
        x_idxs, y_idxs = torch.where(input_ids == 103)

        ## [B, 30552+rels_num]
        logits = out.logits[x_idxs, y_idxs]

        assert logits.shape[0] == out.logits.shape[0]

        if not return_mask_hidden:
            # [B,rels_num]
            return logits[:,-self.rels_num:]
        else:
            # [B,L,768]
            last_hidden_states = out.hidden_states[-1]
            ## [B,768]
            mask_hidden = last_hidden_states[x_idxs, y_idxs]
            assert mask_hidden.shape[0] == last_hidden_states.shape[0]
            ## [B,rels_num], [B,768]
            if not return_cls_hidden:
                return logits[:,-self.rels_num:],mask_hidden
            else:
                cls_hidden = last_hidden_states[:,0,:]
                return  logits[:,-self.rels_num:],mask_hidden,cls_hidden
#     def forward(self, input_ids,attention_mask,token_type_ids,return_mask_hidden=False):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
#                            token_type_ids=token_type_ids,output_hidden_states=return_mask_hidden )

#         ## [MASK] position
#         x_idxs, y_idxs = torch.where(input_ids == 103)

#         ## [B, 30552+rels_num]
#         logits = out.logits[x_idxs, y_idxs]

#         assert logits.shape[0] == out.logits.shape[0]

#         if not return_mask_hidden:
#             # [B,rels_num]
#             return logits[:,-self.rels_num:]
#         else:
#             # [B,L,768]
#             last_hidden_states = out.hidden_states[-1]
#             ## [B,768]
#             mask_hidden = last_hidden_states[x_idxs, y_idxs]
#             assert mask_hidden.shape[0] == last_hidden_states.shape[0]
#             ## [B,rels_num], [B,768]
#             return logits[:,-self.rels_num:],mask_hidden
        
#     def mask_replay_forward(self,input_ids,attention_mask,token_type_ids):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
#                            token_type_ids=token_type_ids,output_hidden_states=True )

#         ## [MASK] position for relation prediction
#         x_idxs_0, y_idxs_0 = torch.where((input_ids==103)*(token_type_ids==0))
#         ## [MASK] position for replay
#         x_idxs_1, y_idxs_1 = torch.where((input_ids == 103) * (token_type_ids == 1))

#         ## [B, 30552+rels_num]

#         logits_0 = out.logits[x_idxs_0, y_idxs_0][:,-self.rels_num:]
#         logits_1 = out.logits[x_idxs_1, y_idxs_1][:,:-self.rels_num]


#         # [B,L,768]
#         last_hidden_states = out.hidden_states[-1]
#         ## [B,768]
#         mask_hidden_0 = last_hidden_states[x_idxs_0, y_idxs_0]
#         mask_hidden_1 = last_hidden_states[x_idxs_1, y_idxs_1]
#         ## [B,rels_num], [B,768]
#         return logits_0,mask_hidden_0,logits_1,mask_hidden_1
    def mask_replay_forward(self, input_ids,attention_mask,token_type_ids,return_mask_hidden=False,return_cls_hidden=False):
        
        B = input_ids.shape[0]
        L = input_ids.shape[1]
        p_randn = torch.rand([B, L]).to(self.device)
        x_mask, y_mask = torch.where((p_randn<self.p_mask)*(token_type_ids == 1))
        mask_token_ids = input_ids[x_mask, y_mask]
        input_ids[x_mask, y_mask] = 103

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids,output_hidden_states=return_mask_hidden )

        ## [MASK] position
        x_idxs, y_idxs = torch.where((input_ids==103)*(token_type_ids==0))

        ## [B, 30552+rels_num]
        logits = out.logits[x_idxs, y_idxs]
        logits_mask = out.logits[x_mask, y_mask]
        loss_mlm = self.mlm_loss_fn(logits_mask,mask_token_ids)
        
        if loss_mlm.cpu().isnan() or self.p_mask<=0:
            loss_mlm = torch.zeros([1])[0].to(self.device)
        
        assert logits.shape[0] == out.logits.shape[0]

        if not return_mask_hidden:
            # [B,rels_num]
            return logits[:,-self.rels_num:],loss_mlm
        else:
            # [B,L,768]
            last_hidden_states = out.hidden_states[-1]
            ## [B,768]
            mask_hidden = last_hidden_states[x_idxs, y_idxs]
            assert mask_hidden.shape[0] == last_hidden_states.shape[0]
            ## [B,rels_num], [B,768]
            if not return_cls_hidden:
                return logits[:,-self.rels_num:],mask_hidden,loss_mlm
            else:
                cls_hidden = last_hidden_states[:,0,:]
                return  logits[:,-self.rels_num:],mask_hidden,cls_hidden,loss_mlm
            
    def mlm_forward(self, mask_hidden):
        #[B,30552+rels_num]
        prediction_scores = self.encoder.cls(mask_hidden)
        #[B,rels_num]
        return prediction_scores[:,-self.rels_num:]
    
    
    def forward_with2mask(self, input_ids,attention_mask,token_type_ids,return_mask_hidden=False,return_cls_hidden=False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids,output_hidden_states=return_mask_hidden )

        ## [MASK] position
        x_idxs, y_idxs = torch.where(input_ids == 103)
        x_idxs = x_idxs.view([-1,2])
        y_idxs = y_idxs.view([-1,2])

        ## [B, 30552+rels_num]
        logits = out.logits[x_idxs[:,1], y_idxs[:,1]]

        task_logits = out.logits[x_idxs[:,0], y_idxs[:,0]]
        assert logits.shape[0] == out.logits.shape[0]

        if not return_mask_hidden:
            # [B,rels_num]
            return logits[:,-self.rels_num:],task_logits[:,1014:1024]
        else:
            # [B,L,768]
            last_hidden_states = out.hidden_states[-1]
            ## [B,768]
            mask_hidden = last_hidden_states[x_idxs[:,1], y_idxs[:,1]]
            assert mask_hidden.shape[0] == last_hidden_states.shape[0]
            ## [B,rels_num], [B,768]
            if not return_cls_hidden:
                return logits[:,-self.rels_num:],mask_hidden,task_logits[:,1014:1024]
            else:
                cls_hidden = last_hidden_states[:,0,:]
                return  logits[:,-self.rels_num:],mask_hidden,cls_hidden,task_logits[:,1014:1024]
            
    def rewrite_samples(self,input_ids,attention_mask,token_type_ids,labels):
        ## Add the relation type information
        x_0, y_0 = torch.where((input_ids==103)*(token_type_ids==0))
        input_ids[x_0, y_0] = labels
        ## mask the sample token
        B = input_ids.shape[0]
        L = input_ids.shape[1]
        p_randn = torch.rand([B, L]).to(self.device)
        x_mask, y_mask = torch.where((p_randn<self.config.rewrite_p)*(token_type_ids == 1))
        input_ids[x_mask, y_mask] = 103
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids,output_hidden_states=True)
        ## [MASK] position
        x_idxs, y_idxs = torch.where((input_ids == 103)*(token_type_ids==1))
        ## [B, 30552+rels_num]
        logits = out.logits[x_idxs, y_idxs][:,:30552]
        ## [N_mask,]
        pred_res = torch.argmax(logits,dim=-1)
        ## restore the mask token
        input_ids[x_idxs, y_idxs] = pred_res
        ## mask the label token
        input_ids[x_0, y_0] = 103
        return input_ids
        
    def ptuning_forward(self, input_ids,attention_mask,token_type_ids,input_embs):
        """
        insert the prompt_emb in the front of the input_ids' word-embedding
        """
        out = self.encoder(input_ids=None,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           inputs_embeds=input_embs,
                           output_hidden_states=True)
        ## [MASK] position
        x_idxs, y_idxs = torch.where(input_ids == 103)
        ## [B, 30552+rels_num]
        logits = out.logits[x_idxs, y_idxs]

        cls_hidden = out.hidden_states[-1][:,0,:]

        assert logits.shape[0] == out.logits.shape[0]
        return logits[:,-self.rels_num:],cls_hidden