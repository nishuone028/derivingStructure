import os
import re
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

# 初始化PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 可以根据需要修改语言

def normalize_text(text):
    """
    对文本进行正则化处理：
    1. 去掉多余的空格
    2. 删除不必要的换行符
    3. 修复中文标点后面没有空格的情况
    4. 修复英文句子中的换行问题
    """
    # b) Add full stop between sentences
    text = re.sub(r'([a-zA-Z0-9 ])(\n+)([A-Z])', r'\1. \2\3', text)
    
    # b) Remove the new line in the middle of a sentence
    text = re.sub(r'([^.])(\n)([^A-Z])', r'\1 \3', text)
    
    # c) Remove duplicated white spaces
    text = re.sub(r' +', ' ', text)

    # d) Clean up multiple newlines
    text = re.sub(r'\n{3,}', r'\n\n', text)

    return text.strip()


def extract_text_from_pdf(pdf_path, output_txt_dir):
    """
    提取PDF中的文本，进行OCR识别，并保存为一个合并的文本文件。
    """
    document = fitz.open(pdf_path)
    full_text = ""
    
    # 获取 PDF 文件的名称（不包括路径和扩展名）用于文件夹和文件命名
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # 创建对应的文件夹（如果不存在）
    pdf_txt_dir = os.path.join(output_txt_dir, pdf_filename)
    os.makedirs(pdf_txt_dir, exist_ok=True)
    
    # 遍历每一页
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        
        # 将PDF页面转为图像
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 将PIL.Image转换为numpy.ndarray
        img_array = np.array(img)

        # 使用PaddleOCR进行文本识别
        result = ocr.ocr(img_array, cls=True)
        
        # 提取并拼接文本
        page_text = ""
        for idx in range(len(result)):
            res = result[idx]
            if res is not None:  # 检查 res 是否为 None
                for line in res:
                    page_text += line[1][0] + " "  # 合并每行文本并用空格分隔
        
        # 对提取的文本进行正则化处理，合并段落
        page_text = normalize_text(page_text)

        full_text += page_text + "\n\n"  # 每页的文本之间用两个换行符分隔

    # 将合并后的完整文本保存为txt文件
    merged_txt_filename = os.path.join(pdf_txt_dir, f'{pdf_filename}_merged.txt')
    with open(merged_txt_filename, 'w', encoding='utf-8') as f:
        f.write(full_text)

    # 返回合并后的完整文本，方便进一步处理
    return full_text

def process_pdf_directory(pdf_directory, output_txt_dir):
    """
    遍历目录中的所有PDF文件，并对每个PDF进行OCR处理。
    """
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"Processing: {pdf_path}")
            extract_text_from_pdf(pdf_path, output_txt_dir)

# 使用示例
pdf_directory = "DATA\\18-23_CSR_REPORT"  # PDF文件目录
output_txt_dir = "output\\txt"    # 保存文本的目录

process_pdf_directory(pdf_directory, output_txt_dir)
