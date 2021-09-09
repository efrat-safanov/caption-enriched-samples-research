#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:58:18 2021

@author: efrat
"""

#This File converts current hateful memes dataset texts to regular text datasets for 
#bert/roberta continued pretraining 

import argparse
import json
import jsonlines
import random

 
   
    
"""
Run with
python convert_data_to_mlm_training.py input_train input_dev output_prefix captions_folder
example: /vol/scratch/efratblaier/datasets/hateful_memes/defaults/image_captions/
"""
def main(args):
    
    dataset_train_texts = []
    dataset_dev_texts = []
    id_to_caption = {}
    
    train_ratio = 0.9
    
    
    def read_captions(file_name, id_to_caption):
        with open(file_name, "r") as filestream:
            for line in filestream:
                currentline = line.split(",")
                if currentline[0] == "id":
                    continue 
                id_to_caption[int(currentline[0])] = currentline[1].strip()
        return True
    
    def add_to_dataset(input_file, id_to_caption, dataset_train_texts, dataset_dev_texts):
        for obj in input_file:
            text = obj["text"]
            if obj["id"] in id_to_caption:
                #text += " [SEP] " + id_to_caption[obj["id"]]
                text += "\n" + id_to_caption[obj["id"]] +"\n"
            print(text)
            if random.random() < train_ratio:
                dataset_train_texts.append(text)
            else: 
                dataset_dev_texts.append(text)
                
    def add_to_dataset_not_random(input_file, id_to_caption, dataset_texts):
        for obj in input_file:
            text = obj["text"]
            if obj["id"] in id_to_caption:
                #text += " [SEP] " + id_to_caption[obj["id"]]
                text += "\n" + id_to_caption[obj["id"]] +"\n"
            dataset_texts.append(text)
    
    if args.captions_folder and args.captions_folder != '':
        read_captions(args.captions_folder + "/image_captioning_dev.csv", id_to_caption)
        read_captions(args.captions_folder + "/image_captioning_test.csv", id_to_caption)
        read_captions(args.captions_folder + "/image_captioning_train.csv", id_to_caption)
    
    random_split = False
    if random_split:
        with jsonlines.open(args.input_train) as input_file:
            add_to_dataset(input_file, id_to_caption, dataset_train_texts, dataset_dev_texts)
        
        with jsonlines.open(args.input_dev) as input_file_dev:
            add_to_dataset(input_file_dev, id_to_caption, dataset_train_texts, dataset_dev_texts)
    
        with jsonlines.open(args.input_test) as input_file_test:
            add_to_dataset(input_file_test, id_to_caption, dataset_train_texts, dataset_dev_texts)
    else:
        with jsonlines.open(args.input_train) as input_file:
            add_to_dataset_not_random(input_file, id_to_caption, dataset_train_texts)
        
        with jsonlines.open(args.input_dev) as input_file_dev:
            add_to_dataset_not_random(input_file_dev, id_to_caption, dataset_dev_texts)
    
        with jsonlines.open(args.input_test) as input_file_test:
            add_to_dataset_not_random(input_file_test, id_to_caption, dataset_train_texts)
    
   
    with open(args.output + "_hateful_memes_for_mlm_train_and_test.txt", mode='w', encoding = 'utf8') as output_file:
        output_file.write("\n".join(dataset_train_texts))
    
    with open(args.output + "_hateful_memes_for_mlm_dev.txt", mode='w', encoding = 'utf8') as output_file_dev:
        output_file_dev.write("\n".join(dataset_dev_texts))

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("input_train", type=str, help="Input train file")
    parse.add_argument("input_dev", type=str, help="Input dev file")
    parse.add_argument("input_test", type=str, help="Input test file")
    parse.add_argument("output", type=str, help="output file prefix")
    parse.add_argument("captions_folder", type=str, help="captions folder")
    args = parse.parse_args()

    main(args)


    
