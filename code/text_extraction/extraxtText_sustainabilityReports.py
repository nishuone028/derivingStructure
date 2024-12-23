from os import makedirs, path
import re
import fitz # PyMuPDF
from pandas import DataFrame
# from ftlangdetect import detect
from langdetect import detect
from collections import defaultdict
from os import path, scandir, listdir
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import opencc

# Extract the text from the PDF document
# INPUT: PDF files 
# OUTPUT: A textual file for each PDF (.pdf --> .txt)

# 初始化PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 可以根据需要修改语言

def extract_text(doc) -> str:
    
    # Iterate over the pages of the document
    documentText = ''
    for page in doc:
        # Get the text from the page
        pageBlocks = page.get_text('blocks', sort = False, flags = fitz.TEXTFLAGS_SEARCH & fitz.TEXT_DEHYPHENATE & ~ fitz.TEXT_PRESERVE_IMAGES)
        
        if pageBlocks:
            pageText = ''
            for block in pageBlocks:
                blockText = block[4] # STRUCTURE: (x0, y0, x1, y1, text, block_no, block_type)

                # a) Remove starting and ending whitespaces
                blockText = blockText.replace('\n', ' ').strip()
                pageText += blockText + '\n'
        else:
            print('\t--> ERROR: No text found in the document,Use OCR instead')
            # 将PDF页面转为图像
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # 将PIL.Image转换为numpy.ndarray
            img_array = np.array(img)

            # 使用PaddleOCR进行文本识别
            result = ocr.ocr(img_array, cls=True)
            
            # 提取并拼接文本
            pageText = ''
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:  # 检查 res 是否为 None
                    for line in res:
                        blockText = line[1][0].replace('\n', ' ').strip()  # 处理每个识别结果
                        pageText += blockText + '\n'  # 添加换行符
        # OTHER FUNCTIONS:  .get_links() // .annots() // .widgets():

        # b) Add full stop in different "sentences" --> e.g., improved readability \n In 2021 the company will ..
        pageText = re.sub(pattern = r'([a-zA-Z0-9 ])(\n+)([A-Z])', repl = r'\1. \2\3', string = pageText)
        
        # b) Remove the new line in the middle of a sentence --> e.g., improved readability \n of the code
        pageText = re.sub(pattern = r'([^.])(\n)([^A-Z])', repl = r'\1 \3', string = pageText)
            
        # c) Remove duplicated white spaces --> e.g., improved  readability
        pageText = re.sub(pattern = r' +', repl = ' ', string = pageText)

        # Save the page text in the document text --> pages separted by two new lines 
        documentText += pageText + '\n\n'
    
    # d) Remove duplicated page separators
    documentText = re.sub(pattern = r'\n{3,}', repl = r'\n\n\n', string = documentText)
    
    return documentText.strip()

def remove_single_char_lines(text):
    # 使用正则表达式匹配形如 \nX\n 的模式，其中 X 是单个字符
    return re.sub(r'\n(.)\n', lambda m: '\n' if len(m.group(1)) == 1 else m.group(0), text)

def remove_extra_spaces(text):
    # 使用正则表达式替换多余的空格为单个空格
    return re.sub(r'\s+', ' ', text).strip()

def format_text(text):
    """
    This function removes newline characters from the text and segments it based on periods.
    It then adds a newline character after each segment.
    """
    # Remove newline characters
    text_without_newlines = text.replace('\n', '')
    
    # Segment the text based on periods
    segments = text_without_newlines.split('。')
    
    # Add a newline character after each segment
    formatted_text = '。\n'.join(segments)
    
    return formatted_text


def remove_lines_without_any_punctuation(text):
    # 标点符号的正则表达式，包括中文和英文标点
    punctuation = r'[\u3000-\u303F.,;!?，。；！？]'
    
    # 按行分割文本
    lines = text.split('\n')
    
    lines = [remove_extra_spaces(line) for line in lines]
    # 过滤掉不包含任何标点符号的行
    filtered_lines = [line for line in lines if re.search(punctuation, line)]
    
    # 将过滤后的行重新组合成字符串
    return '\n'.join(filtered_lines)

def remove_special_characters(text):
    # 定义一个字符串，包含所有需要去除的特殊字符
    special_chars = "■●□√☆"
    # 使用字符串的translate方法去除特殊字符
    return text.translate(str.maketrans('', '', special_chars))

# 判断文本是否是繁体
def is_traditional(text):
    traditional_range = r'[\u4e00-\u9fff\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F]'
    return bool(re.search(traditional_range, text))

# 创建繁体转简体的转换器
cc = opencc.OpenCC('t2s')  # 繁体转简体

def extract_text_with_formatting(doc):
    full_text = ""
    for page in doc:
        text_instances = page.get_text()  # Get all text instances on the page
        
        if is_traditional(text_instances):
            text_instances = cc.convert(text_instances)

        if text_instances:
        
            text_instances = remove_single_char_lines(text_instances) # Remove single-character lines

            text_instances = remove_lines_without_any_punctuation(text_instances)

            text_instances = remove_special_characters(text_instances)
        
            text_instances = format_text(text_instances)

        else:
            print('\t--> ERROR: No text found in the document,Use OCR instead')
            # 将PDF页面转为图像
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # 将PIL.Image转换为numpy.ndarray
            img_array = np.array(img)

            # 使用PaddleOCR进行文本识别
            result = ocr.ocr(img_array, cls=True)
            
            # 提取并拼接文本
            text_instances = ''
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:  # 检查 res 是否为 None
                    for line in res:
                        blockText = line[1][0].replace('\n', ' ').strip()  # 处理每个识别结果
                        text_instances += blockText + '\n'  # 添加换行符

            text_instances = remove_single_char_lines(text_instances) # Remove single-character lines

            text_instances = remove_lines_without_any_punctuation(text_instances)

            text_instances = remove_special_characters(text_instances)

            text_instances = format_text(text_instances)

        full_text += text_instances
    return full_text.strip()

def text_loader(report_data, analyze_language = True):

    for companyName, reports in report_data.items():
        print('\nCOMPANY:', companyName)
        
        for idk, url_report in enumerate(reports):

            # Read the PDF file
            try:
                with fitz.open(url_report['path']) as doc:    # type: ignore     
                    
                    
                    # document_text = extract_text(doc)
                    
                    document_text = extract_text_with_formatting(doc)
                    # Ensure a good utf-8 encoding
                    document_text = document_text.encode('utf-8', errors = 'replace').decode('utf-8')
                    
                    # Check if the text is duplicated
                    is_duplicatedText = any([doc['text'] == document_text for doc in report_data[companyName] if 'text' in doc.keys() and doc['text'] != None])
                    if is_duplicatedText:
                        print("Duplicate text")
                        continue
                    
                    # Save the number of pages
                    report_data[companyName][idk]['numPages'] = doc.page_count           
                    
                    # Extract the text from the document
                    report_data[companyName][idk]['text'] = document_text
                    
                    # Extract the language
                    if analyze_language:
                        # predicted_language = detect(text = document_text.replace("\n"," "),  low_memory = False)
                        predicted_language = detect(text = document_text.replace("\n"," "))
                        # report_data[companyName][idk]['language'] = predicted_language['lang']
                        report_data[companyName][idk]['language'] = predicted_language
                    
                    print(f'--> REPORT {idk +1}/{len(reports)}: ' \
                        f"[{doc.metadata['format']}| PAGES:{doc.page_count}] '{url_report['documentType']}'")

            except RuntimeError as runtimeError:
                print(f'\t--> ERROR: {runtimeError}')
                report_data[companyName][idk]['text'] = None
                
    return report_data


def save_textualData(report_data, saving_folder):
    for companyName, reports in report_data.items():
        for report in reports:

            if 'text' not in report.keys() or report['text'] == None:
                continue

            # Create the folder for the year
            year_folder = path.join(saving_folder, report['year'])

            if not path.exists(year_folder):
                makedirs(year_folder)
                
            # File name 
            if 'language' in report.keys():
                fileName = companyName + "-" + report["documentType"] + "-" + report['language'].upper() + ".txt"
            else:
                fileName = companyName + "-" + report["documentType"] + ".txt"

            # Save the textual data
            with open(path.join(year_folder, fileName), mode = 'w', encoding = 'utf-8') as txt_file:
                
                try:
                    txt_file.write(report['text'])
                except UnicodeEncodeError as unicodeError:
                    print(f'\t--> ERROR: {unicodeError}')
                    report['text'] = report['text'].encode('utf-8', errors = 'replace').decode('utf-8')
                    txt_file.write(report['text'])
                    
def numPages_stats(saving_folder, report_data):
    df = DataFrame([report['numPages'] for reports in report_data.values() for report in reports 
                    if 'numPages' in report.keys()], columns=['numPages'])

    df = df.describe()    
    df.to_excel(path.join(saving_folder, 'numPages.xlsx'))
    
    return df

def documentMetadata_loader(folderPath):
    reports = defaultdict(list)
    for yearlyFolder in scandir(folderPath):
        if not yearlyFolder.is_dir():
            print('A file was found in the folder', folderPath, 'and it was ignored')
            continue

        # Extract information
        year = yearlyFolder.name
        files = listdir(yearlyFolder.path)
        for fileName in files:
   
            #fileParts = fileName.split('.')
            #documentName = fileParts[0].strip() if len(fileParts) <= 2 else ''.join(fileParts[:-1])
            #fileExtension = fileParts[-1].strip()
            documentName, fileExtension  = path.splitext(fileName)

            if fileExtension.lstrip('.') not in ['pdf', 'txt', 'html', 'xhtml']:
                print('\nWARNING! Wrong extension for file "', fileName, '" in folder', yearlyFolder.path, 'and it was ignored')
                continue

            # Extract company name and document type
            partialComponents = documentName.split('_')
            
            if len(partialComponents) < 2 or len(partialComponents) > 3:
                raise Exception("\n" + documentName + "-->" + str(partialComponents) + '\nThe file name <<'+ fileName + '>> is not in the correct format!<companyName>-<documentType>[-<documentLanguage>]')

            # Parse the company name and document type
            partialComponents = [re.sub(pattern = r' +', repl = ' ', string = component.strip()) for component in partialComponents]
            companyName = partialComponents[0]
            documentType = partialComponents[1]
            
            # Parse the document language
            if len(partialComponents) == 3:
                documentLanguage = partialComponents[2]
            else:
                documentLanguage = None

            # Save information
            reports[companyName].append({
                'year': year,
                'documentType': documentType,
                'documentLanguage': documentLanguage,
                'fileExtension': fileExtension,
                'path' : path.join(yearlyFolder.path, fileName)})

    reports = dict(sorted(reports.items(), key = lambda dict_item: dict_item[0]))

    return reports
                   

if __name__ == '__main__':


    # Folder paths for the reports
    data_path = 'E:\\derivingStructure\ori_data'
    rawData_path = path.join(data_path, '18-23_CSR_REPORT')
    # rawData_path = path.join(data_path, 'test')
    # Load the paths of the reports
    report_data = documentMetadata_loader(rawData_path)

    # Load the textual data
    report_data = text_loader(report_data)

    # Save the textual data (i.e., extracted texts)
    saving_folder = path.join(data_path, 'processed', 'nonFinancialReports' + '_REGEX')
    save_textualData(report_data, saving_folder)
    
    # Saving the stats
    save_numPages = numPages_stats(path.join(data_path, 'processed'), report_data)