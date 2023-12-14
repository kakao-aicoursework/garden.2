from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain, LLMRouterChain
from langchain.prompts.chat import ChatPromptTemplate

from dto import ChatbotRequest
from samples import list_card

import os
import aiohttp
import time
import logging

CUR_DIR = os.path.dirname(os.path.abspath('/Users/garden/dev/github/garden.2/llm_garden/project_02/data/'))
BUG_STEP1_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/bug_say_sorry.txt")
BUG_STEP2_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/bug_request_context.txt")
ENHANCE_STEP1_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/enhancement_say_thanks.txt")
INTENT_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "data/parse_intent.txt")
INTENT_LIST_TXT = os.path.join(CUR_DIR, "data/intent_list.txt")

# 환경 변수 처리 필요!
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

def generate_answer(user_message: str) -> str:
    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)

    llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")
    parse_intent_chain = create_chain(
        llm=llm,
        template_path=INTENT_PROMPT_TEMPLATE,
        output_key="intent",
    )
    bug_step1_chain = create_chain(
        llm=llm,
        template_path=BUG_STEP1_PROMPT_TEMPLATE,
        output_key="bug-step1",
    )
    bug_step2_chain = create_chain(
        llm=llm,
        template_path=BUG_STEP2_PROMPT_TEMPLATE,
        output_key="bug-step2",
    )
    enhance_step1_chain = create_chain(
        llm=llm,
        template_path=ENHANCE_STEP1_PROMPT_TEMPLATE,
        output_key="enhance-step1",
    )
    default_chain = ConversationChain(llm=llm, output_key="text")

    intent = parse_intent_chain(context)["intent"]

    if intent == "bug":
        answer = ""
        for step in [bug_step1_chain, bug_step2_chain]:
            answer += step.run(context)
            answer += "\n\n"
    elif intent == "enhancement":
        answer = enhance_step1_chain.run(context)
    else:
        answer = OpenAI(temperature=0.9)(user_message)
    return answer

async def callback_handler(request: ChatbotRequest) -> dict:

    user_message = request.userRequest.utterance
    output_text = generate_answer(user_message)

   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }

    # debug
    # print(output_text)

#     # ===================== end =================================
#     # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
#     # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, ssl=False) as resp:
                await resp.json()