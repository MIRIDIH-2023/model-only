import json
import logging
import os
import random

from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset

from core.common.utils import img_trans_torchvision, get_visual_bbox
from core.datasets.collate_supervised import DataCollatorForT5DocCLS
from core.datasets.collator_self_supervised import DataCollatorForSelfSupervisedTasks

import pandas as pd

EMPTY_BOX = [0, 0, 0, 0]
SEP_BOX = [1000, 1000, 1000, 1000]

logger = logging.getLogger(__name__)


def normalText(t):
    if type(t) is float:
        if t == int(t):
            t = int(t)
    t = str(t)
    return t.strip()


def get_prop(node, name):
    title = node.get("title")
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


def get_bb(bb):
    bbs = [float(j) for j in bb]
    xs, ys = [], []
    for i, b in enumerate(bbs):
        if i % 2 == 0:
            xs.append(b)
        else:
            ys.append(b)
    return [min(xs), min(ys), max(xs), max(ys)]


def get_rvlcdip_labels():
    return [
        "letter",
        "form",
        "email",
        "handwritten",
        "advertisement",
        "scientific report",
        "scientific publication",
        "specification",
        "file folder",
        "news article",
        "budget",
        "invoice",
        "presentation",
        "questionnaire",
        "resume",
        "memo"
    ]

def random_masking(L=4096, mask_ratio=0.75):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(L)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=0)

    # keep the first subset
    ids_keep = ids_shuffle[:len_keep]
    ids_remove = ids_shuffle[len_keep:]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([L])
    mask[:len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=0, index=ids_restore)
    return mask, ids_restore, ids_remove, ids_keep

# Argument : random_masking의 mask
# Returns : masking할 곳의 slice들
def group_tokens(mask):

    group_lst = []
    i=0
    prev=0

    for m in mask:
        if m == 0:
            if i == prev:
                pass
            else:
                group_lst.append([prev, i])
            prev = i+1
        i += 1

    if prev != i:
        group_lst.append([prev, i])

    return group_lst

# Argument : ori_bbox_lst, group_tokens의 리턴 list (slices)
# Returns : masking된 부분의 그룹화된 bbox
def group_bbox(bbox_lst, group_lst):

    bbox_group_lst = []

    for s in group_lst:
        target = bbox_lst[s[0]:s[1]]
        if len(target) == 1:
            bbox_group_lst.append(*target)
        else:
            t = target[0][0]
            l = target[0][1]
            b = target[0][2]
            r = target[0][3]
            for i in target[1:]:
                if i[0] < t:
                    t = i[0]
                if i[1] < l:
                    l = i[1]
                if i[2] > b:
                    b = i[2]
                if i[3] > r:
                    b = i[3]
            bbox_group_lst.append([t,l,b,r])
    
    return bbox_group_lst

# Argument : ori_bbox_list, mask_ratio
# Returns : token slices to be masked, grouped bboxes
def mask_process(bbox_list, mask_ratio=0.75):
    l = len(bbox_list)
    mask = random_masking(L=l, mask_ratio=mask_ratio)
    return group_tokens(mask[0]), group_bbox(bbox_list, group_tokens(mask[0]))

class RvlCdipDataset(Dataset):

    #NUM_LABELS = 16

    def __init__(self , xml_sample_loc , json_sum_loc, image_path, tokenizer , data_args , mode='train', user_prompt=None):

        """ Structure of data directory:

            --- xml_sample_loc (.csv)
                   ├── images_url
                   └── labels_url
            --- data (folder)
                   └── processed_sample{index} .json
        """
        self.main_df = pd.read_csv(xml_sample_loc) # xml_sample.csv 파일 저장
        self.image_path = image_path

        with open(json_sum_loc, 'r', encoding='utf8') as f:
            self.main_json_data = json.load(f)

        if mode == 'train': #train ,val, test 에 따라 사용하는 data의 범위가 다름. (근데 self-supervised도 이거 필요 있나..? )
            file_data_range = ( 0 , int(len(self.main_df) * 0.6 ) )
        elif mode == 'val':
            file_data_range = ( int(len(self.main_df) * 0.6 ) , int(len(self.main_df) * 0.8 ) )
        elif mode == 'test':
            file_data_range = ( int(len(self.main_df) * 0.8 ) , int(len(self.main_df) ) )
        else:
            raise NotImplementedError

        self.cls_bbox = EMPTY_BOX[:]
        self.pad_bbox = EMPTY_BOX[:]
        self.sep_bbox = SEP_BOX[:]

        self.tokenizer = tokenizer
        self.max_seq_length = data_args.max_seq_length
        self.num_img_embeds = 0

        label_list = get_rvlcdip_labels() #classification task에서만 사용하는 변수
        self.label_list = label_list
        self.label_map = dict(zip(list(range(len(self.label_list))), self.label_list))
        self.n_classes = len(label_list)
        self.label_list = label_list

        self.image_size = data_args.image_size

        self.examples = []
        self.labels = []
        self.images = []

        self.cls_collator = DataCollatorForSelfSupervisedTasks( #기존에 정의한 토크나이저 선언
                  tokenizer=tokenizer,
            )
        
        self.user_prompt = user_prompt

        results = [self.load_file(file_idx) for file_idx in tqdm(range(file_data_range[0],file_data_range[1]))]
        for labels, examples, images in results:
            self.labels += labels
            self.examples += examples
            self.images += images
        
        assert len(self.labels) == len(self.examples)

    def load_file(self, file_idx):

        labels = []
        examples = []
        images = []

        labels.append(0) ############### label 미정으로 일단 다 0 ##########
        examples.append(file_idx)
        images.append(file_idx)

        return labels, examples, images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index): #완료
        try:
            label = self.labels[index]
            label = self.label_map[int(label)]

            upt=''
            if self.user_prompt is None:
                r = random.randint(0,1)
                if r == 0:
                    upt = 'Layout Modeling.'
                elif r == 1:
                    upt = 'Visual Text Recognition.'
                else:
                    upt = 'Joint Text-Layout Reconstruction'
            else:
                upt = self.user_prompt

            rets, n_split = read_ocr_core_engine(json_data = self.main_json_data , user_prompt=upt, image_path = self.image_path , file_idx = index , tokenizer = self.tokenizer, max_seq_length = self.max_seq_length, num_img_embeds = self.num_img_embeds, image_size = self.image_size)
            if n_split == 0:
                # Something wrong with the .ocr.json file
                print(f"EMPTY ENTRY in index {index}")
                return self[(index + 1) % len(self)]
            for i in range(n_split): #정상적으로 코드 실행됬다면 n_split==1 임.
                text_list, bbox_list, labels, image, page_size = rets[i]
                (width, height) = page_size
                bbox = [  #이미지 크기에 맞게 정규화
                    [
                        b[0] / width,
                        b[1] / height,
                        b[2] / width,
                        b[3] / height,
                    ]
                    for b in bbox_list
                ]

                visual_bbox_input = get_visual_bbox(self.image_size) # (x_min, y_min, x_max, y_max) 형태의 좌표로 이루어진 텐서 반환

                #input_ids = self.tokenizer.convert_tokens_to_ids(text_list) #토큰 자른것들을 token id들로 변환

                #input_ids, labels, bbox_input = self.cls_collator("user prompt", text_list, bbox, label) #prompt 붙여서 최종 input,bbox,label을 만듦. ################################
                input_ids, labels, bbox_input = text_list, labels, bbox_list

                attention_mask = [1] * len(input_ids)
                decoder_attention_mask = [1] * len(labels)

                char_list = [0]
                char_bbox_list = [[0,0,0,0]]
                char_ids = torch.tensor(char_list, dtype=torch.long)
                char_bbox_input = torch.tensor(char_bbox_list, dtype=torch.float)

                bbox_input = torch.tensor(bbox_input, dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.long)
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)
                assert len(bbox_input) == len(input_ids)
                assert len(bbox_input.size()) == 2
                assert len(char_bbox_input.size()) == 2

                return_dict =  {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "seg_data": bbox_input,
                    "visual_seg_data": visual_bbox_input,
                    "decoder_attention_mask": decoder_attention_mask,
                    "image": image,
                    'char_ids': char_ids,
                    'char_seg_data': char_bbox_input
                }
                assert input_ids is not None

                return return_dict
        except: #오류가 났다는 거는 파일이 없다는 것. 해당 상황에서는 index+1 파일 불러오는 것으로 대체
            #image는 로딩 중 오류 생긴 파일 그냥 해당 index가 없게 저장해서 문제 없음.
            #json파일도 오류 생긴건 해당 index없어서 걸러짐.
            return self[(index + 1) % len(self)]

    #def get_labels(self): # classification에서 label의 종류 출력하는 함수. 우리는 필요 없을 듯.
    #    return list(map(str, list(range(self.NUM_LABELS))))

    def pad_tokens(self, input_ids, bbox): #이건 그냥 길이 max_len에 맞게 맞추는 함수
        # [CLS], sentence, [SEP]
        tokenized_tokens = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        start_token, _, end_token = tokenized_tokens[0], tokenized_tokens[1:-1], tokenized_tokens[-1]

        sentence = tokenized_tokens
        expected_seq_length = self.max_seq_length - self.num_img_embeds
        mask = torch.zeros(expected_seq_length)
        mask[:len(sentence)] = 1

        bbox = [self.cls_bbox] + bbox + [self.sep_bbox]
        while len(sentence) < expected_seq_length:
            sentence.append(self.tokenizer.pad_token_id)
            bbox.append(self.pad_bbox)

        assert len(sentence) == len(bbox)
        return (sentence, mask, bbox, start_token, end_token)

# 해당 부분은 json파일의 문장을 line-by-line으로 읽는 것으로 해당 함수 수정 완료.
def read_ocr_core_engine(json_data, user_prompt, image_path, file_idx, tokenizer, max_seq_length=None, num_img_embeds=None, image_size=224):
    #max_seq_length와 num_img_embeds 는 원본 코드에서도 안쓰는데 왜있는거지?

    # Labeling할 토큰들을 정한다
    if 'Layout Modeling' in user_prompt:
        mask_ratio = 0.75
    elif 'Visual Text Recognition' in user_prompt:
        mask_ratio = 0.5
    elif 'Joint Text-Layout Reconstruction' in user_prompt:
        mask_ratio = 0.15
    else :
        raise ValueError('Invalid Prompt')

    #file_ 와 image_dir 모두 1 2 3 ... index 임.    
    data = json_data[file_idx] # 하나로 합쳐진 json파일에서 현재 idx 가져옴.

    rets = []
    n_split = 0

    #page_size = (1280,720)
    page_size = data['form'][0]['sheet_size']

    tiff_images =  Image.open(f"{image_path}/image_{file_idx}.png")
    tiff_images.convert('RGB')

    image = img_trans_torchvision(tiff_images, image_size)

    collator = DataCollatorForSelfSupervisedTasks(
       tokenizer = tokenizer,
    )

    sub_text_list, sub_bbox_list, labels_list = [], [], []
    ret_text_list, ret_bbox_list = [], []
    for form in data['form']: #문장별로 쪼갬
      text_list, bbox_list = [], []
      for word in form['words']: #단어별로 쪼갬

        if word == ' ': #띄어쓰기는 건너뛰기
          continue

        sub_tokens = tokenizer.tokenize(word['text']) #단어별로 쪼갠걸 다시 토큰화 (하나의 단어도 여러개의 토큰 가능)
        for sub_token in sub_tokens:
          text_list.append(sub_token)
          bbox_list.append(word['box']) #현재는 단어별 bbox, 추후 문장별 bbox로도 수정 가능
          #bbox_list.append(form['box'])
      sub_text_list.append(text_list)
      sub_bbox_list.append(bbox_list)

    assert len(sub_text_list) == len(sub_bbox_list)

    a = 0
    for i in range(len(sub_text_list)):

        group_list, group_bbox_list = mask_process(sub_bbox_list[i], mask_ratio=mask_ratio)

        b = a + len(group_list)
        numbering_list = [i%100 for i in range(a,b)]
        a = b

        # range를 토대로 numbering list를 만든다
        ids_list = tokenizer.convert_tokens_to_ids(sub_text_list[i])

        # 변수 설명
        # user_prompt = 말 그대로 user_prompt
        # ids_list = 한 문장의 token들을 index(id)로 변환한 것들을 모아놓은 리스트 (그룹화 안됨)
        # bbox_list = 한 문장의 token들에 대응하는 bounding box를 모아놓은 리스트(그룹화 안됨)
        # group_list = masking할 범위를 slice (e.g. [a,b]) 형태로 저장해 놓은 것들의 리스트 (그룹화 됨)
        # group_bbox_list = group_list에 대응하는, 즉 그룹화 된 것들에 대응하는 bounding box를 모아놓은 리스트 (그룹화 됨)
        # numbering_list = sentinel_token에 번호를 부여하기 위해 넘겨주는 리스트
        # page_size = 이미지 가로세로 (bbox 정규화하여 라벨링할 때 사용)
        # 왜 collator 안에서 bbox를 정규화함??
        #  -> 우리가 만든 rvlcdip 데이터셋 클래스 안에서 read_ocr(이 함수)에서 받은 bbox 정규화를 하기 때문에
        #  여기서 bbox를 정규화하면 좀 귀찮아지기 때문에 collator 안에서 하는 게 좋다 
        #
        # Collator 안에서 이 변수들을 활용하여 labeling 및 sentinel token을 붙인 후 리턴하시면 됩니다
        # Collator에서 labeling할 때에는 ids_list, 혹은 bbox_list를 기준으로 iteration을 하되
        # group_list를 보면서 만약 index가 group_list에 있는 slice들중 하나에 해당된다, 하면은
        # sentinel token을 붙이고, slice에 포함된 범위는 masking한 뒤 labeling을 적절히 하시면 되겠습니다
        input_ids, labels, bbox_list = collator(user_prompt, ids_list, sub_bbox_list[i], group_list, group_bbox_list, numbering_list, tiff_images.size)
        ret_text_list += input_ids
        ret_bbox_list += bbox_list
        labels_list += labels
    
    prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
    length = len(prompt_ids)
    ret_text_list = prompt_ids + ret_text_list
    ret_bbox_list = [[0,0,0,0]] * length + ret_bbox_list

    if len(text_list) > 0:
      ret_text_list += [1] # </s> token
      ret_bbox_list.append([0,0,0,0]) # </s> token's bbox
      labels_list += [1] # </s> token
      rets.append([ret_text_list, ret_bbox_list, labels_list, image, page_size])

    assert len(ret_text_list) == len(ret_bbox_list)
    n_split = len(rets)

    return rets, n_split