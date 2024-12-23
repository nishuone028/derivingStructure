import os
import re

def extract_company_name(filename):
    # 使用正则表达式匹配公司名称
    match = re.match(r'(.+?)-\d{4}年度社会责任报告', filename)
    if match:
        return match.group(1)
    return None

def list_unique_companies(directory):
    companies = set()
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                company_name = extract_company_name(file)
                if company_name:
                    companies.add(company_name)
                    
    return list(companies)

# 示例用法
directory_path = 'E:\\derivingStructure\\ori_data\\processed\\nonFinancialReports_REGEX'
unique_companies = list_unique_companies(directory_path)
print(len(unique_companies))
print(unique_companies)