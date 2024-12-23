
import itertools
import pandas as pd
import torch.cuda as cuda
import torch
import os

from os import  path
from tqdm import tqdm
from more_itertools import batched
from psutil import Process
from time import perf_counter, strftime
from kor import create_extraction_chain
from json import load
from langchain_community.chat_models import ChatZhipuAI, ChatOpenAI

# LOCAL IMPORTS
from utils.document_loader import NonFinancialDisclosuresDataset, SemanticSearcher
from models.wizardLM_langchain import WizardLM, GLM4CausalLM
from models.prompt.srl_schema import schema, prompt_template

def load_chatgpt():
    # Load the language model via API
    print("\nLoading the language model via API...")
    api_key = 
    base_url = 
    llm_model = ChatOpenAI(
        model_name='gpt-3.5-turbo', 
        api_key=api_key, 
        base_url=base_url,
        temperature=0,
        max_tokens=2000,
        model_kwargs={
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'top_p': 1.0
        }
    )
    return llm_model

def load_chatglm():
    # Load the language model via API
    print("\nLoading the language model via API...")
    zhipuai_chat = ChatZhipuAI(
    temperature=0,
    api_key=,
    model="glm-4",
    max_tokens=2000,
    top_p=1.0
    # api_base="...",
    # other params...
    )
    return zhipuai_chat


if __name__ == '__main__':
    print("PROCESS PID:", Process().pid)
    print('DEVICE:', '(GPU) ' + cuda.get_device_name() if cuda.is_available() else 'CPU')
    
    # Read settings
    # with open(file = path.join('end2end_KG', 'genSRL', 'configs.json')) as json_file:
    file='E:\\derivingStructure\\code\\end2end_KG\\genSRL\\configs_zh.json'
    with open(file, 'r', encoding='utf-8') as json_file:
        configs = load(json_file)
        params = configs['params']
        selected_companies = configs['companies_setTEST']
        
        if params['debug']:
            selected_companies = selected_companies[:2]
      
    # Start the counter
    program_start = perf_counter()
    timing_stats = dict()
    
    # data_path = path.join('..', '..','data')
    data_path = 'E:\\derivingStructure\\ori_data'
    nonFinancialFiles_path = path.join(data_path, 'Processed', 'nonFinancialReports_REGEX')
    
    dataset = NonFinancialDisclosuresDataset(
        data_path = nonFinancialFiles_path, 
        languages = params['document_languages'],
        selected_companies = selected_companies, 
        select_most_recent_companyDoc = params['select_most_recent_companyDoc'],
        sentence_minWords = params['sentence_minWords'], 
        sentences_per_input = params['sentences_per_input'],
        max_sentences = 30 if params['debug'] else -1
    )
    
    print(len(dataset))
    if params['sentence_filtering']['topk'] != -1:
        semantic_searcher = SemanticSearcher()
        selected_indices = semantic_searcher.filter_sentences(
            sentences = dataset, 
            sentence_per_keyword = params['sentence_filtering']['topk'], 
            sim_threshold = params['sentence_filtering']['sim_threshold'], 
            verbose = True)
        dataset = dataset.select_subset(selected_indices)
        
        del semantic_searcher
        cuda.empty_cache()
        
    print("CUDA:", str(round(cuda.max_memory_allocated() * (10 ** -9), 2)) + ' GB')
    

    # Load the language model
    print("\nLoading the language model...")
    # llm_model = WizardLM()
    # llm_model = GLM4CausalLM(
    #     model_path="THUDM/glm-4-9b-chat",
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16
    # )

    llm_model = load_chatglm()
    
    print("Dataset loaded:", dataset)

    # 2) Initialize the LLM chain
    chain = create_extraction_chain(llm_model, schema, encoder_or_encoder_class='json') #, instruction_template=prompt_template) 
    print("*" * 100 + "\n" + "*" * 100 + "\n" + "*" * 100 + "\n", 
          chain.prompt.format_prompt(text="[user input]").to_string(), 
          "\n" + "*" * 100 + "\n" + "*" * 100 + "\n" + "*" * 100, "\n")    
    
    # Get the model outputs
    #data_loader = DataLoader(dataset, batch_size = params['batch_dim'], num_workers = 0,
                            #collate_fn = lambda sentences: [sentence for sentence in sentences]) 
                            
    # outputs = itertools.chain.from_iterable(
    #     chain.apply(batch) for batch in tqdm(batched(dataset, n = params['batch_dim']),  # type: ignore
    #                                          desc = "batches", total = len(dataset) // params['batch_dim']))
    
    # Initialize an empty list for storing all extracted triples
    all_outputs = []

    # Process the documents in batches
    print(f"Computing {len(dataset)} documents...")
    start_processing = perf_counter()

    # Iterate over the dataset in batches
    for batch in batched(dataset, n=params['batch_dim']):
        for text_item in batch:
            print("text_item:", text_item['text'])
            # Apply the chain to the batch
            result = chain.invoke(text_item)  # Apply the LLM chain to the batch
            print("text_item result:", result)  # Debug: Print batch result
        
            # Flatten the result and add it to the all_outputs list
            all_outputs.append(result)
    print('all_outputs:', all_outputs)
# text_item result: {'text': {'data': {'esg_actions': [{'esg_category': '员工福利与关怀', 'predicate': '组织、提供、发放、设立', 'object': '中医服务、心理咨询和疏导服务、女性健康讲座、慰问品、专用座厕', 'properties': {'sub_esg_category': '员工福利活动', 'agent': '南宁分行、黑龙江分行、信用卡中心、客户满意中心、广州分行', 'time': '3月8日', 'location': '广州分行办公楼', 'purpose': '员工提供福利与关怀'}}]}, 'raw': '<json>{"esg_actions": [{"esg_category": "员工福利与关怀", "predicate": "组织、提供、发放、设立", "object": "中医服务、心理咨询和疏导服务、女性健康讲座、慰问品、专用座厕", "properties": {"sub_esg_category": "员工福利活动", "agent": "南宁分行、黑龙江分行、信用卡中心、客户满意中心、广州分行", "time": "3月8日", "location": "广州分行办公楼", "purpose": "员工提供福利与关怀"}}]}</json>', 'errors': [], 'validated_data': {}}}
# document_loader报错。。
    # Save the triples into the dataset
    if all_outputs:  # Ensure there is data to save
        dataset.store_triples(list_triples=all_outputs)
    else:
        print("No triples were extracted from the dataset.")

    # Stop the timer
    timing_stats['sentence_processing'] = perf_counter() - start_processing
    timing_stats['total'] = perf_counter() - program_start
    time_elapsed = divmod(timing_stats['total'], 60)

    
    # Save the dataset on the disk
    saving_folder = path.join('outputs', 'genSRL')
    # Ensure the saving folder exists
    os.makedirs(saving_folder, exist_ok=True)
    file_name = 'triples_chatGLM_' + params['test_name']
    dataset.save(saving_folder, file_name, sort_triples = True if not params['debug'] else False)
    
    # Save some technical stats
    stats_df = pd.Series({
        'timestamp': strftime("%Y-%m-%d, %H:%M"),
        'script_params': params,
        'inputs':{
            'num_docs': dataset.stats['documents_loaded'],
            'companies': ', '.join(dataset.stats['companies']),
            'total_sentences': len(dataset),
            'sentence_coverage': dataset.stats['sentence_coverage']
        },
        'run_stats':{
            'computational_time': {
                'total': f'{round(time_elapsed[0])} minutes and {round(time_elapsed[1])} seconds',
                'sentence_processing': timing_stats['sentence_processing'],
                'avg_sentence': round(timing_stats['sentence_processing'] / len(dataset), 1)
            },
            'cuda':{
                'max_memory_allocated': str(round(cuda.max_memory_allocated() * (10 ** -9), 2)) + ' GB',
                'max_memory_reserved': str(round(cuda.max_memory_reserved() * (10 ** -9), 2)) + ' GB',
            }
        },
        'model_params':llm_model._identifying_params
    })
    
    with open(path.join(saving_folder, file_name + '_stats.json'), mode = 'w', encoding = 'utf-8') as json_file:
        stats_df.to_json(json_file, orient = 'index', indent = 4, force_ascii = False)
        
    print("||| END |||")