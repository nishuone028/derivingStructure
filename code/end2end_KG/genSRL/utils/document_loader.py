from collections import defaultdict
import torch

from more_itertools import batched
from pysbd import Segmenter
from os import path
from torch.utils.data import Dataset, DataLoader, Subset
import jieba
import re

from pandas import Series
from numpy import array, concatenate
from sentence_transformers.util import cos_sim
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings ,FakeEmbeddings
from numpy import array

# LOCAL IMPORTS
import sys
sys.path.insert(0, path.join(path.dirname(__file__), '..', '..', '..'))
from _library.data_loader import loader_nonFinancialReports, load_esg_categories
# from utils.toolkits import parse_output, attach_tripleProperties


from ujson import loads, JSONDecodeError
import dirtyjson

def parse_output(output):
    # 提取 raw 字段中的字符串
    raw_text = output['text']['raw']
    output = raw_text.lstrip('<json>').rstrip('</json>')

    # output = output['text'].lstrip('<json>').rstrip('</json>')
    
    try:
        parsed_output = loads(output)
    except JSONDecodeError as parsing_error:
        try:
            parsed_output = dirtyjson.loads(output)
        except dirtyjson.error.Error:
            print("FAILED PARSING:\n", output)
            return []
        
    if isinstance(parsed_output, dict) and 'esg_actions' in parsed_output.keys():
        return parsed_output['esg_actions']
    
    return []

def parse_output_chatgpt(output):
    # 检查 'data' 字典中是否存在 'esg_actions' 键，并且它对应的值是一个非空列表
    if 'data' in output['text'] and 'esg_actions' in output['text']['data'] and output['text']['data']['esg_actions']:
        # 检查 'errors' 列表是否为空
        if 'errors' in output['text'] and output['text']['errors']:
            print("Errors found, skipping parsing.")
            return []

        # 提取 raw 字段中的字符串
        raw_text = output['text']['raw']
        # 去除 <json> 和 </json> 标签
        output_str = raw_text.strip('<json>').strip('</json>')
        
        try:
            parsed_output = dirtyjson.loads(output_str)
            # 检查解析后的数据是否为字典，并且包含 'esg_actions' 键
            if isinstance(parsed_output, dict) and 'esg_actions' in parsed_output:
                return parsed_output['esg_actions']
        except dirtyjson.error.Error as parsing_error:
            print(f"FAILED PARSING: {parsing_error}\n{output_str}")
    
    return []

def attach_tripleProperties(df_row):
    to_add = {'source': str(df_row['year']) + '-' + df_row['documentName'], 'original_sentence': df_row['text']}
                                                                         
    for triple in df_row['triples']:
        if len(triple.keys()) > 0:
            if 'properties' in triple.keys():
                triple['properties'].update(to_add)
            else:
                triple['properties'] = to_add
    return df_row['triples']


class NonFinancialDisclosuresDataset(Dataset):
    def __init__(self, data_path, languages, selected_companies, select_most_recent_companyDoc, sentence_minWords, sentences_per_input, max_sentences = -1):
        
        self.languages = languages
        self.stats = dict()
        
        # (1) Load the documents
        print("\nNonFinancialDisclosuresDataset")
        self.docs = self._load_documents(data_path, languages, selected_companies)
        
        if select_most_recent_companyDoc:
            num_docs = len(self.docs)
            
            # Sort the documents by year
            self.docs = self.docs.sort_values(by = ['companyName', 'year'], ascending = [True, False])
            print(self.docs)
            # Drop duplicates
            self.docs = self.docs.drop_duplicates(subset = 'companyName', keep = 'first').reset_index(drop = True)
            print(self.docs)
            print(f"\n[INFO] Dropped {num_docs - len(self.docs)} documents since there the companies have more than one documents, and the most recent ones have been selected.\n")
        
        print(f"\t--> Document selected ({len(self.docs)}) by {len(self.stats['companies'])} companies (" + ', '.join(self.stats['companies']) + ')')
        print('\t\t-->', '\n\t\t--> '.join( f"[{doc['companyName']}] {doc['documentName']}" for doc in self.docs[['companyName', 'documentName']].to_dict(orient = 'records')))
        
        # (2) Segment documents
        self._segment_docs(sentence_minWords, sentences_per_input)
        print("--> Documents have been segmented")
        
        # (3) DEBUG: subset of sentences
        if max_sentences != -1:
            print(len(self.docs))
            max_sentences = min(max_sentences, len(self.docs))

        if max_sentences != -1:
            self.docs = self.docs.sample(max_sentences, random_state = 101)
            print("--> [DEBUG] Sentence subset:", len(self.docs))
        
        # Random order
        self.docs =  self.docs.sample(frac = 1, random_state = 101).reset_index(drop = True)
        
        # Save the inputs
        #.map(lambda text: Document(page_content = text)).to_numpy() 
        self.inputs = array(self.docs[['text', 'companyName']].to_dict(orient = 'records')) # type: ignore
        self.stats['inputs'] = len(self.inputs)
    
    def select_subset(self, indices):
        self.docs = self.docs.iloc[indices]
        
        self.inputs = self.docs['text'].to_frame().to_dict(orient = 'records') # type: ignore
        self.stats['inputs'] = len(self.inputs)
        
        return self
        
    def store_triples(self, list_triples):
        self.docs['triples'] = Series(
            # data = (parse_output(item) for item in list_triples),
            data = (parse_output_chatgpt(item) for item in list_triples),
            index = self.docs.index)
        
        # Attach two attributes to the triples (source and original sentence)
        self.docs['triples'] = self.docs.apply(attach_tripleProperties, axis = 1)
        
        self.stats['sentence_coverage'] = round(len(self.docs[self.docs['triples'].str.len() > 0]) / len(self.docs), 4)

    def save(self, saving_folder, file_name, sort_triples = True):
        
        # Grouping
        outcome_df = self.docs[['companyName', 'triples']].groupby(by = 'companyName').agg(lambda triples: concatenate(array(triples)).ravel())
        
        if sort_triples:
            # Order triples according to the ESG category & predicate
            try:
                outcome_df = outcome_df['triples'].map(lambda triples: sorted(triples, key = lambda triple: (triple['esg_category'], triple['predicate']))).sort_index()
            except KeyError as error:
                print(error)
                print(outcome_df)
                pass
        
        print("\nOUTCOMES:\n", outcome_df)
        
        # Save to dataframe as a JSON file
        with open(path.join(saving_folder, file_name + '.json'), mode = 'w', encoding = 'utf-8') as json_file:
            outcome_df.to_json(json_file, orient = 'index', indent = 4, force_ascii = False)
            
        print(f"\n||| Outcomes have been saved! ({file_name})|||\n")
        
    def _load_documents(self, data_path, languages, selected_companies):
        
        # Read the documents
        print("Loading documents...")
        docs = loader_nonFinancialReports(data_path, companies = selected_companies, read_text = True).drop_duplicates(subset = 'text')
        
        # Filter the documents based on the languages
        if languages:
            docs = docs[docs['documentLanguage'].str.lower().isin(languages)]
        print(f"--> Documents have been loaded ({len(docs)}, {'|'.join(map(str.upper, languages))})")
        
        self.stats['total_documents'] = len(docs)
        
        # Select companies
        #selected_companies = [company.lower() for company in selected_companies]
        #docs = docs[docs['companyName'].str.lower().isin(selected_companies)].reset_index(drop = True)
        
        self.stats['documents_loaded'] = len(docs)
        self.stats['companies'] = docs['companyName'].unique()
        
        return docs
    
    def _segment_docs(self, sentence_minWords, sentences_per_input):
        sentenceSegmenter = Segmenter(language = self.languages[0], clean = True, doc_type = "pdf")
        
        self.docs['text'] = self.docs['text'].map(sentenceSegmenter.segment)
        
        # 记录原始句子数量
        original_sentences = self.docs['text'].explode().tolist()
        original_sentence_count = len(original_sentences)
        
        print(sentence_minWords)
        # 根据字符数或词数过滤句子
        def filter_sentences(sentences):
            return tuple(batched(
                iterable = (sentence for sentence in sentences 
                            if len(self._filter_words(jieba.lcut(sentence))) >= sentence_minWords),  # 过滤标点符号后按词数过滤
                n = sentences_per_input
            ))
        self.docs['text'] = self.docs['text'].map(lambda sentences: filter_sentences(sentences))
        
        # 记录过滤后的句子数量
        filtered_sentences = self.docs['text'].explode().tolist()
        filtered_sentence_count = len(filtered_sentences)
        
        # 计算被删除的句子
        deleted_sentences = set(original_sentences) - set(filtered_sentences)
        
        # 打印被删除的句子
        print(f"\n[INFO] Deleted {original_sentence_count - filtered_sentence_count} sentences:")
        for sentence in deleted_sentences:
            print(f"- {sentence}")
            print(f"  len(self._filter_words(jieba.lcut(sentence))): {len(self._filter_words(jieba.lcut(sentence)))}")
            print(f"  self._filter_words(jieba.lcut(sentence)): {self._filter_words(jieba.lcut(sentence))}")

        # Explode the sentences: an entry for each document
        self.docs = self.docs.explode(column = "text")
        
        # Merge the sentences in each batch
        self.docs['text'] = self.docs['text'].str.join(' ')
        self.docs = self.docs.drop_duplicates(subset = 'text')
    
    # 过滤标点符号的函数
    def _filter_words(self, words):
        # 只保留中文和英文字母的词，去除标点符号
        return [word for word in words if re.match(r'^[a-zA-Z0-9\u4e00-\u9fa5]+$', word)]

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.inputs[idx]

class SemanticSearcher:
    def __init__(self) -> None:
        #self.model = FakeEmbeddings(size = 1024)
        
        print('\n[SemanticSearcher] Loading the embedding model...')
        self.model = HuggingFaceInstructEmbeddings(
            model_name = "E:\\derivingStructure\\instructor_models", 
            model_kwargs = {'device': 'cuda'},
            query_instruction = "Represent the title for retrieving relevant statements: ", # Financial
            embed_instruction = "Represent the statement for retrieval: ")
        
        # SGPT-2.7
        #self.model = HuggingFaceEmbeddings(model_name = "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit", model_kwargs = {'device': 'cuda'})
    
    def filter_sentences(self, sentences, keywords = None, sentence_per_keyword = 1, sim_threshold = 0.5, verbose = False):
        
        # Define the default keywords
        if not keywords:
            keywords = load_esg_categories()['Environmental']
            
        if sentence_per_keyword > len(sentences):
            sentence_per_keyword = sentence_per_keyword // len(keywords)
        if sentence_per_keyword < 1:
            sentence_per_keyword = 1
        
        # Generate the embeddings for the keywords ()
        keyword_embeddings = torch.Tensor([self.model.embed_query(text = keyword) for keyword in keywords])
        
        # Generate the embeddings for all the sentences
        sentence_embeddings = torch.Tensor([self.model.embed_documents([sentence['text'] for sentence in sentences])]).reshape(len(sentences),-1)
        
        company_sentences = defaultdict(list)
        for idk, sentence in enumerate(sentences):
            company_sentences[sentence['companyName']].append(idk)
        
        print("\n[KEYWORDS] Embeddings:", str(keyword_embeddings.size()))
        print("[SENTENCES] Embeddings:", str(sentence_embeddings.size()))
        
        # For each keyword
        selected_indices = set()
        for keyword, embedded_keyword in list(zip(keywords, keyword_embeddings)):
            for companyName in set(company_sentences.keys()):
                print("\n" + '-' * 10, f"KEYWORD: {keyword} || COMPANY: {companyName} --> docs: {len(company_sentences[companyName])}",  '-' * 10)

                # Select the sentences of the company
                company_sentence_embedding = sentence_embeddings[company_sentences[companyName]]
            
                # Compute the similarities between the embedded keyword and the sentence embeddings
                similarities = cos_sim(a = company_sentence_embedding, b = embedded_keyword).flatten()
                
                if sim_threshold != -1:
                    indices_to_skip = torch.argwhere(similarities < sim_threshold)
                    
                    print(f"Similarities ({len(similarities)}): MAX: {similarities.max()} | MIN: {similarities.min()} || "\
                        f"ABOVE SIM THRESHOLD ({sim_threshold * 100} %):", len(similarities) - len(indices_to_skip), f'({round(((len(similarities) - len(indices_to_skip)) / len(similarities)) * 100, 2)} %)')
                else:
                    indices_to_skip = []
                # Select the top k documents
                if sentence_per_keyword > len(similarities):
                    sentence_per_keyword = len(similarities)
                
                topk_indices = torch.topk(similarities, k = sentence_per_keyword).indices.tolist()
                
                # Remove indices that are under the threshold
                if sim_threshold != -1:
                    topk_indices = [index for index in topk_indices if index not in indices_to_skip]
                
                if len(topk_indices) > 0:
                    
                    # Mapping the indices to the original indices
                    similarity_idx = topk_indices
                    topk_indices = [company_sentences[companyName][index] for index in topk_indices]
                    
                    # Save the indices
                    selected_indices.update(topk_indices)
                    
                    if verbose:
                        print(f"TOP{len(topk_indices)} documents:")
                        print(('\n' + '*' * 50 + '\n').join(f'{idk +1}) [{round(sim.item(), 2)}] ' + sentence['text'] for idk, (sentence, sim) in enumerate(zip(sentences[topk_indices], similarities[similarity_idx]))), '\n')
                else:
                    print('There are no sentences above the threshold.')
        
        filtering_stats = {companyName: len([idx for idx in indices if idx in selected_indices]) for companyName, indices in company_sentences.items()}
        print(f"\nFILTERED SENTENCES: {len(selected_indices)}/{len(sentences)} ({round((len(selected_indices) / len(sentences)) * 100, 0)} %)")
        print(dict(sorted(filtering_stats.items(), key = lambda item: item[1], reverse = True)))    
            
        return list(selected_indices)