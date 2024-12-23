from langchain_community.chat_models import ChatOpenAI,ChatZhipuAI
from kor import create_extraction_chain, Object, Text 

# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo",
#     api_key = 'idr_0979b98556658331aa90452d0b4fb46c',
#     base_url = 'https://apiok.us/api/openai/v1',
#     temperature=0,
#     max_tokens=2000,
#     model_kwargs = {
#         'frequency_penalty':0,
#         'presence_penalty':0,
#         'top_p':1.0
#     }
# )


def load_chatglm():
    # Load the language model via API
    print("\nLoading the language model via API...")
    zhipuai_chat = ChatZhipuAI(
    temperature=0,
    api_key="853d98925735ba42ee0deaf1a7e53077.vx7mC90SEwbkH3j5",
    model="glm-4",
    max_tokens=2000,
    top_p=1.0
    # api_base="...",
    # other params...
    )
    return zhipuai_chat


schema = Object(
    id="player",
    description=(
        "User is controlling a music player to select songs, pause or start them or play"
        " music by a particular artist."
    ),
    attributes=[
        Text(
            id="song",
            description="User wants to play this song",
            examples=[],
            many=True,
        ),
        Text(
            id="album",
            description="User wants to play this album",
            examples=[],
            many=True,
        ),
        Text(
            id="artist",
            description="Music by the given artist",
            examples=[("Songs by paul simon", "paul simon")],
            many=True,
        ),
        Text(
            id="action",
            description="Action to take one of: `play`, `stop`, `next`, `previous`.",
            examples=[
                ("Please stop the music", "stop"),
                ("play something", "play"),
                ("play a song", "play"),
                ("next song", "next"),
            ],
        ),
    ],
    many=False,
)

llm = load_chatglm()
chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
result = chain.invoke("play songs by paul simon and led zeppelin and the doors")
print('result:',result)

# from ujson import loads, JSONDecodeError
# import dirtyjson

# text_item  =  {'text': {'data': {'esg_actions': [{'esg_category': '员工福利与关怀', 'predicate': '组织、提供、发放、设立', 'object': '中医服务、心理咨询和疏导服务、女性健康讲座、慰问品、专用座厕', 'properties': {'sub_esg_category': '员工福利活动', 'agent': '南宁分行、黑龙江分行、信用卡中心、客户满意中心、广州分行', 'time': '3月8日', 'location': '广州分行办公楼', 'purpose': '员工提供福利与关怀'}}]}, 'raw': '<json>{"esg_actions": [{"esg_category": "员工福利与关怀", "predicate": "组织、提供、发放、设立", "object": "中医服务、心理咨询和疏导服务、女性健康讲座、慰问品、专用座厕", "properties": {"sub_esg_category": "员工福利活动", "agent": "南宁分行、黑龙江分行、信用卡中心、客户满意中心、广州分行", "time": "3月8日", "location": "广州分行办公楼", "purpose": "员工提供福利与关怀"}}]}</json>', 'errors': [], 'validated_data': {}}}

# def parse_output_chatgpt(output):
#     # 提取 raw 字段中的字符串
#     raw_text = output['text']['raw']
#     output = raw_text.lstrip('<json>').rstrip('</json>')

#     # output = output['text'].lstrip('<json>').rstrip('</json>')
    
#     try:
#         parsed_output = loads(output)
#     except JSONDecodeError as parsing_error:
#         try:
#             parsed_output = dirtyjson.loads(output)
#         except dirtyjson.error.Error:
#             print("FAILED PARSING:\n", output)
#             return []
        
#     if isinstance(parsed_output, dict) and 'esg_actions' in parsed_output.keys():
#         return parsed_output['esg_actions']
    
#     return []

# data = parse_output_chatgpt(text_item)
# print(data)

