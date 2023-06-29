import math
import collections
import pickle
import os
from random import shuffle

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

# TODO: 각각의 Task에 따라 라벨링 하기
# 근데 Dataset에서 input text가 id로 변환되어가지고
# 개인적인 생각으로는 다시 token으로 변환하던가, 아니면 decode까지 해서 하던가 해야 할 듯??
# 그냥 Dataset에서 따로 라벨링을 하는 게 좀 더 편할 것 같긴 한디...

# Wrapper Class
# User Prompt에 따라서 Collator를 호출해주는 클래스입니다
class DataCollatorForSelfSupervisedTasks:

    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):
        
        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

        self.LM = DataCollatorForT5LayoutModeling(
            tokenizer = self.tokenizer,
            input_length = self.input_length,
            target_length = self.target_length,
            pad_token_id = self.pad_token_id,
            decoder_start_token_id = self.decoder_start_token_id
        )

        self.VT = DataCollatorForT5VisTextRec(
            tokenizer = self.tokenizer,
            input_length = self.input_length,
            target_length = self.target_length,
            pad_token_id = self.pad_token_id,
            decoder_start_token_id = self.decoder_start_token_id
        )

        self.JR = DataCollatorForT5JointReconstruction(
            tokenizer = self.tokenizer,
            input_length = self.input_length,
            target_length = self.target_length,
            pad_token_id = self.pad_token_id,
            decoder_start_token_id = self.decoder_start_token_id
        )

    # Label Numbering 변수 추가.
    # label_numbering 변수가 하는 일
    #   1. Sentinel Token에 번호를 부여한다
    #   2. 엥 근데 이렇게 구현하면 User_Prompt가 여러 개 포함되는 거 아님?
    #   3. label_numbering에 0이 없으면 user_prompt를 안넣으면 된다.
    # label_numbering 변수의 형태 = [a, b]
    # a = 시작, b = 끝
    # mask_process는 read_ocr_core_engine에서 불러서 mask할 범위를 집어 줄 것이다
    def __call__(self, user_prompt, ori_input_ids, bbox_list, group_list, ori_bbox_list, label_numbering, page_size):

        if 'Layout Modeling' in user_prompt:
            ret = self.LM(ori_input_ids, bbox_list, group_list, ori_bbox_list, label_numbering, page_size)
        
        elif 'Visual Text Recognition' in user_prompt:
            ret = self.VT(ori_input_ids, bbox_list, group_list, ori_bbox_list, label_numbering, page_size)
        
        elif 'Joint Text-Layout Reconstruction' in user_prompt:
            ret = self.JR(user_prompt, ori_input_ids, bbox_list, group_list, ori_bbox_list, label_numbering, page_size)
        
        else:
            raise ValueError("Invalid user prompt")

        return ret

# Sub Class에서 해야 할 일
# 1. Argument를 user_prompt, ori_input_ids, group_list, ori_bbox_list, label_numbering을 받는다.
# 2. label_numbering의 시작이 0이라면, user_prompt를 앞에 id로 변환해서 붙인다. (input_ids에)
# 3. label_numbering에 따라서 sentinel token을 ori_bbox_list를 보면서 붙인다. (input_ids에)
# 4. labeling은 group_list와 ori_bbox_list를 보면서 붙인다 (labels에)
class DataCollatorForT5LayoutModeling:
    """
    Data collator used for T5 Layout Modeling
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, input_ids, bbox_list, group_list, group_bbox_list, label_numbering, page_size):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box
        
        res_input_ids = []
        res_bbox_list = []

        labels = []
        for i in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_l_id_{label_numbering[i]}>', add_special_tokens=True)[:-1]
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][0]/page_size[0])}>')[:-1]
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][1]/page_size[1])}>')[:-1]
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][2]/page_size[0])}>')[:-1]
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][3]/page_size[1])}>')[:-1]
            
        slice_pointer=0
        L = len(group_list)
        input_len = len(input_ids)
        for i in range(input_len):
            if slice_pointer < L and i == group_list[slice_pointer][0]:
                temp_ids = self.tokenizer.encode(f'<extra_l_id_{label_numbering[slice_pointer]}>', add_special_tokens=True)[:-1]
                res_input_ids += temp_ids
                res_input_ids.append(input_ids[i])
                res_bbox_list += [[0,0,0,0]] * len(temp_ids)
                res_bbox_list.append(bbox_list[i])
            elif slice_pointer < L and i == group_list[slice_pointer][1] :
                temp_ids = self.tokenizer.encode(f'</extra_l_id{label_numbering[slice_pointer]}>', add_special_tokens=True)[:-1]
                res_input_ids += temp_ids
                res_input_ids.append(input_ids[i])
                res_bbox_list += [[0,0,0,0]] * len(temp_ids)
                res_bbox_list.append(bbox_list[i])
                slice_pointer += 1
            else:
                res_input_ids.append(input_ids[i])
                res_bbox_list.append(bbox_list[i])
                
        if slice_pointer < L and input_len == group_list[slice_pointer][1] :
            temp_ids = self.tokenizer.encode(f'</extra_l_id{label_numbering[slice_pointer]}>', add_special_tokens=True)[:-1]
            res_input_ids += temp_ids
            res_bbox_list += [[0,0,0,0]] * len(temp_ids)
        
        return res_input_ids, labels, res_bbox_list

class DataCollatorForT5VisTextRec:
    """
    Data collator used for T5 Visual Text Recognition
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    # Sub Class에서 해야 할 일
    # 1. Argument를 user_prompt, ori_input_ids, group_list, ori_bbox_list, label_numbering을 받는다.
    # 2. label_numbering의 시작이 0이라면, user_prompt를 앞에 id로 변환해서 붙인다. (input_ids에)
    # 3. label_numbering에 따라서 sentinel token을 ori_bbox_list를 보면서 붙인다. (input_ids에)
    # 4. labeling은 group_list와 ori_bbox_list를 보면서 붙인다 (labels에)
    def __call__(self, input_ids, bbox_list, group_list, group_bbox_list, label_numbering, page_size):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box
        tmp_input_ids = []
        tmp_bbox_list = []

        # TODO: 라벨링 하기 (Visual_Text_Recognition_Test.py 참고하면서)
        labels = []
        for i in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_t_id_{label_numbering[i]}>', add_special_tokens=True)[:-1]
            labels += input_ids[group_list[i][0]:group_list[i][1]]


        slice_pointer=0
        L = len(group_list)
        for i in range(len(input_ids)):
            if slice_pointer < L and i == group_list[slice_pointer][0]:
                tmp_input_ids += self.tokenizer.encode(f'<extra_t_id_{label_numbering[slice_pointer]}>', add_special_tokens=True)[:-1]
                #print(f'extra : {len(self.tokenizer.encode(f"<extra_t_id_{label_numbering[slice_pointer]}>", add_special_tokens=True)[:-1])}')
                tmp_bbox_list.append([0,0,0,0])
                bbox_ids = []
                for j in range(4):
                    if j % 2 == 1:
                        bbox_ids += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[slice_pointer][j]/page_size[1])}>', add_special_tokens=True)[:-1]
                        #print(f'loc : {len(self.tokenizer.encode(f"<loc_{int(500*group_bbox_list[slice_pointer][j]/page_size[1])}>", add_special_tokens=True)[:-1])}')
                    else:
                        bbox_ids += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[slice_pointer][j]/page_size[0])}>', add_special_tokens=True)[:-1]
                        #print(f'loc : {len(self.tokenizer.encode(f"<loc_{int(500*group_bbox_list[slice_pointer][j]/page_size[0])}>", add_special_tokens=True)[:-1])}')
                    tmp_bbox_list.append([0,0,0,0])
                tmp_input_ids += bbox_ids
                tmp_input_ids += self.tokenizer.encode(f'</extra_t_id_{label_numbering[slice_pointer]}>', add_special_tokens=True)[:-1]
                #print(f'extra : {len(self.tokenizer.encode(f"</extra_t_id_{label_numbering[slice_pointer]}>", add_special_tokens=True)[:-1])}')
                tmp_bbox_list.append([0,0,0,0])
                i = group_list[slice_pointer][1]-1
                slice_pointer += 1
            else:
                tmp_input_ids.append(input_ids[i])
                tmp_bbox_list.append(bbox_list[i])

        return tmp_input_ids, labels, tmp_bbox_list


class DataCollatorForT5JointReconstruction:
    """
    Data collator used for T5 Joint Text-Layout Reconstruction
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, user_prompt , input_ids, bbox_list, group_list, group_bbox_list, label_numbering, page_size):
        
        prompt_text = user_prompt
        length = 0
        tmp_input_ids = []
        if label_numbering[0] == 0:
            prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
            length = len(prompt_ids)
            tmp_input_ids = prompt_ids
            tmp_bbox_list = [[0,0,0,0]] * length
        else:
            length = 0
            tmp_input_ids = []
            tmp_bbox_list = []

        # TODO: 라벨링 하기 (Visual_Text_Recognition_Test.py 참고하면서)
        labels = []
        for i in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_id_{label_numbering[i]}>', add_special_tokens=False)
            labels += input_ids[group_list[i][0]:group_list[i][1]]
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][0]/page_size[0])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][1]/page_size[1])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][2]/page_size[0])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][3]/page_size[1])}>', add_special_tokens=False)

        slice_pointer=0
        L = len(group_list)
        for i in range(len(input_ids)):
            if slice_pointer < L and i == group_list[slice_pointer][0]:
                tmp_input_ids += self.tokenizer.encode(f'<extra_id_{label_numbering[slice_pointer]}>', add_special_tokens=False)
                tmp_bbox_list.append([0,0,0,0])

                i = group_list[slice_pointer][1]-1
                slice_pointer += 1
            else:
                tmp_input_ids.append(input_ids[i])
                tmp_bbox_list.append(bbox_list[i])

        return tmp_input_ids, labels, tmp_bbox_list
    
