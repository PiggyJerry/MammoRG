import os
current_dir = os.path.dirname(os.path.abspath(__file__))
class Config(object):
    def __init__(self):

        self.max_len = 512
        self.rel_num = 3
        self.bert_max_len = 1024
        self.bert_dim = 768
        self.tag_size = 4  # For HB-TB, HB-TE, HE-TE and others

        self.checkpoint = f'{os.path.dirname(current_dir)}/checkpoint/OneRel.pt'