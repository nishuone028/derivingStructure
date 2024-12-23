from kor import Object, Text, Number
from kor.nodes import Number, Object, Option, Selection, Text

from langchain import PromptTemplate
from os import path

# LOCAL IMPORTS
import sys
sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))
from _library.data_loader import load_esg_categories

    
labelled_sentences = [
    (
        "集团根据实际业务变化，每年动态更新《新城控股反贿赂与反腐败制度》；制定并实施《商业行为准则》《反贿赂与反腐败制度》《审计监察管理办法》《利益冲突管理办法》《员工职务行为准则》《员工手册》《投诉举报管理办法》《礼品礼金管理办法》等管理制度和规范，推动各类监督贯通协同，保障集团内部反腐倡廉规范化、体系化。",
        [
            {
                "esg_category": "反腐败", 
                "predicate": "更新并实施", 
                "object": "《商业行为准则》《反贿赂与反腐败制度》 《审计监察管理办法》《利益冲突管理办法》《员工职务行为准则》《员工手册》《投诉举报管理办法》《礼品礼金管理办法》等管理制度和规范",
                "properties" : {
                    "sub_esg_category": "反商业贿赂及反贪污",
                    "agent": "新城控股",
                    "time": "每年",
                    "manner": "推动和保障",
                    "purpose": "各类监督贯通协同、内部反腐倡廉规范化、体系化"
                }
            }
        ]
    ),( 
     "国投集团高度重视各利益相关方的关切和诉求，丰富与利益相关方的沟通渠道，优化与利益相关方的沟通方式，通过打造国投集团融媒体平台，实现信息共享沟通，积极回应利益相关方对国投集团的履责期待，努力为利益相关方创造更大价值。",
     [
          {
                "esg_category": "利益相关沟通", 
                "predicate": "高度重视、积极回应", 
                "object": "关切和诉求",
                "properties" : {
                    "sub_esg_category": "外部利益相关方沟通",
                    "agent": "国投集团",
                    "manner": "打造、实现",
                    "purpose": "国投集团融媒体平台、信息共享沟通、努力为利益相关方创造更大价值"
                }
          }
        ] 
    ),( 
     "华润燃气强化监督检查、隐患整治，督促落实总经理定期安全检查制度，配合华润集团开展要素化审核、安全检查，共发现隐患 ２４０项，截至 ２０２２年 １２月 ３１日，已全部完成整改，整改率 １００％。",
     [      
            {
                "esg_category": "职业健康与安全生产", 
                "predicate": "强化、督促落实、配合", 
                "object": "总经理定期安全检查制度",
                "properties" : {
                    "sub_esg_category": "安全审计和检查", 
                    "agent": "华润燃气",
                    "location": "３８家成员单位",
                    "time": "截至2022年12月31日",
                    "manner": "开展",
                    "purpose": "监督检查、隐患整治；要素化审核、安全检查"
                }
            }
        ] 
    ), (
        "中国三星大力开展资源综合利用，最大程度实现废物资源化和再生资源回收利用，减少垃圾焚烧产生的温室气体和污染物，防止垃圾掩埋造成的土壤和地下水污染。",
        [ 
         {
                "esg_category": "循环经济", 
                "predicate": "大力开展", 
                "object": "垃圾焚烧产生的温室气体和污染物",
                "properties" : {
                    "sub_esg_category": "废物资源化和再生资源回收利用",
                    "agent": "中国三星",
                    "manner": "最大程度实现、防止",
                    "purpose": "资源综合利用、废物资源化和再生资源回收利用、垃圾掩埋造成的土壤和地下水污染"
                }
            }
        ]
    ),(
        "项目选址执行生态保护红线、环境质量底线、资源利用上线和环境准入负面清单要求，避让自然保护地及野生动物重要栖息地、迁徙洄游通道等，开展项目区域内动植物及其栖息地的调查，明确生物多样性保护措施。",
        [{
                "esg_category": "生态系统与生物多样性保护", 
                "predicate": "执行、避让", 
                "object": "自然保护地及野生动物重要栖息地",
                "properties" : {
                    "sub_esg_category": "生态系统保护", 
                    "manner": "开展",
                    "purpose": "生态保护红线、环境质量底线、资源利用上线和环境准入负面清单要求"
                }
            }
        ]
    )
]

schema = Object(
    id="esg_actions",
    description="与企业环境、社会或治理方面的相关行动",
    attributes=[
        Selection(
            id="esg_category",
            description="与ESG方面相关的问题",
            options=[Option(id = category_name.lower().replace(' ', '_').replace('-', '_'), description = category_name) 
                     for category_name in load_esg_categories()['Environmental']]
        ),
        Text(
            id="predicate",
            description="在ESG（环境、社会和治理）相关领域中产生影响的动词形式",
        ),
        Text(
            id="object",
            description="与ESG（环境、社会和治理）类别相关的实体，它受到谓语动作的影响或作用"
        ),
        Object(
            id = "properties",
            description = '描述主题（通常指句子或讨论的中心内容）、谓语（表示主题所进行的行为或状态）和宾语（接受行为或状态的对象）的特性或属性',
            attributes = [
                Text(
                    id="sub_esg_category",
                    description = "与环境、社会和治理（ESG）中的一个特定方面相关的具体问题"
                ),
                Text(
                    id="agent",
                    description = "一个有意地执行某项行为的参与者"
                ),
                Text(
                    id="location",
                    description = "行为发生的地点"
                ),
                Text(
                    id="time",
                    description = "行为发生的时间"
                ),
                Text(
                    id="manner",
                    description = "行为被执行的方法或方式",
                ),
                Text(
                    id="purpose",
                    description = "执行谓语（句子中的动词部分）的原因或目标"
                )  
            ]
        )],
    many = True,
    examples = labelled_sentences
)

prompt_template = PromptTemplate(
    input_variables=["type_description", "format_instructions"],
    template=(
        "您的任务是从用户的输入中提取符合以下所示模式的结构化信息。在提取过程中，请确保信息与模式中的数据类型完全一致。请勿添加模式中未列出的任何属性。\n\n"
        "{type_description}\n\n"
        "{format_instructions}\n\n"
    )
)

prompt_template_ = PromptTemplate(
    input_variables=["type_description", "format_instructions"],
    template=(
        "请根据下面描述任务的指令和提供的上下文输入，写出一个恰当的回复来完成这个请求。\n\n### 指令:\n"
        "您的任务是从用户的输入中提取符合以下所示模式的结构化语义信息。在提取过程中，您可以考虑句法关系，但必须保持属性和值之间的语义一致性。请确保每个句子的界限清晰，并独立处理每个句子，避免在不同句子间混淆语义信息。在提取信息时，请确保它与模式中的信息类型完全一致，不要添加模式中未列出的任何属性。\n\n"
        "{type_description}\n\n"
        "{format_instructions}\n\n### 输入:\n"
    )
)