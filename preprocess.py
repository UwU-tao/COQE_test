import torch
import json
import numpy as np
import re
import random
import os

def convert_data(data_type):
    folder_path = ''
    des_file = ''
    if data_type == 'train':
        folder_path = './data/smartphone/VLSP2023_ComOM_training_v2'
        des_file = './data/smartphone/train.txt'
    if data_type == 'test':
        folder_path = './data/smartphone/VLSP2023_ComOM_testing_v2'
        des_file = './data/smartphone/test.txt'
    if data_type == 'dev':
        folder_path = './data/smartphone/VLSP2023_ComOM_dev_v2'
        des_file = './data/smartphone/dev.txt'


    files = os.listdir(folder_path)

    sentences_and_content = []
  
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            sections = file.read().split('\n\n')

            for section in sections:
                tmp=""
                parts = section.split('\n')
                if len(parts) >= 2:
                    sentence = parts[0].strip()
                    tmp, sentence = sentence.split('\t')
                    sentence = " ".join(sentence.split())
                    sentence += '\t' + '1'
                    sentences_and_content.append(sentence)
                else:
                    sentence = parts[0].split('\t')
                    if len(sentence) == 1:
                        continue
                    else:
                        tmp, sentence = sentence
                    sentence = " ".join(sentence.split())
                    sentence += '\t' + '0'
                    sentences_and_content.append(sentence)
    
    with open(des_file, 'w', encoding='utf-8') as output_file:
        for item in sentences_and_content:
            output_file.write(str(item) + '\n')

convert_data('train')
convert_data('test')
convert_data('dev')