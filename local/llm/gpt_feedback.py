# -*- coding: utf-8 -*-
# @Time       : 11/17/2024
# @Author     : Yu-Hsuan Fang (Original version created by him; thanks a lot!)
# @Affiliation: National Taiwan Normal University and EZAI

import json
import openai
import os
import logging
import time
import yaml
import re
#logging.setLevel(logging.WARNING)

class GenerateText:
    def __init__(self, conf="./conf/llm_config.yaml") -> None:

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        
        self.config = {}
        self.message_list = []
        
        with open(conf) as f:
            self.config = yaml.safe_load(f)

        openai.api_key = self.config["OPENAI_API_KEY"]
        self.model = self.config["model"]
        
        with open(self.config["feedback_guideline"], "r") as fn:
            guideline_content = fn.read()
            
        self.message_list.append({"role": "system", "content": guideline_content})
        self.message_list.append({"role": "user", "content" : "有一個八維的發音屬性向量，每一維向量代表口型的位置數據，如第一維度，代表jaw（下巴）的位置，使用數字表示所處的位置：0:Nearly Closed：1:Neutral, 2:Slightly Lowered，如第一維度是0，表示jaw(下巴）近乎關閉，以此類推。第二維度lip separation：0:Closed, 1: Slightly Apart, 2:Apart, 3:Wide Apart。 第三維度lip rounding： 0:Rounded, 1:Slightly Rounded,  2:Neutral,  3:Spread。 第四維度tongue frontness: 0:Back, 1:Slightly Back, 2:Neutral, 3:Slightly Front。第五維度tongue height：0:Low, 1:Mid, 2:Mid-High, 3:High。第六維度tongue tip: 0:low, 1: mid, 2: mid-High 3: High 第七維度velum：0:Closed, 1:Open。第八維度voicing，0:Unvoiced, 1:Voiced。現在需要你依據語音學和生理學的知識，通過計算兩個發音屬性向量（一个為正確的發音屬性向量，一個為實際說話時的發音屬性向量）的相似度，來判斷說話人是否有發音錯誤，如果有請給出原因和建議。例如發IPA phone b時，正確發音屬性向量為（1，0，2，2，1，1，0，1）或（1，2，2，2，1，1，0，1），說話人發音屬性向量為（1，0，2，2，1，1，0，0），要求你通過計算發音向量的相似度，通過語音學的知識判斷該學習者是否發音錯誤並給出理由。"})
        self.message_list.append({"role": "user", "content" : "音素中的sil為靜音標籤，不需考慮作為音素的比較，可以將它視作語氣停頓與詞句轉換的起始。"})
        self.message_list.append({"role": "user", "content" : "音素轉發音屬性的關係如下：sh 22133000，iy 01343001 ，hh 22221100，ae 33230001，d  11243401， 12242301，y  12243301，er 22221001，aa 32110001，r  12122301，k  12203100 ，12202100，s  12233300，uw 11013001，t  112434001， 2242300，ih 32342001，n  11223411，g  12203101 ，12202101，w  12003101，ao 32102001，dh 22242201，l  12232401，ow 32132001 ，21012001，m  10221111，eh 32231001，oy 22101001 ，12332001，ay 32220001， 12332001，b  10221101， 12221101，v  20221101，f  20221100，z  12233301，th 22242200，ah 22220001，p  10221100 ，12221100 ，ey 12342001，ng 12203111，ch 22243400， 12133000，uh 12112001，zh 22133001，jh 22243401， 12133001，aw 32220001， 12012001。請你記住這張映射表，每個音素對應8個數字（8維發音屬性向量），或者16個數字（兩個8維發音屬性向量）。"})

        self.request_template = f'''請輸出對L2學習者的回饋，
跟讀文本為：<PROMPT_WORDS>。
對應的正確發音為：<PROMPT_PHONES>。
實際的發音為: <PREDICT_PHONES>。
請先對正確發音和實際發音進行分詞，
然後找到每個word對應的發音，
接著通過對比每個word的phone sequence，
找到其中存在的替換，刪除，插入錯誤，
然後對錯誤地方，給出有實際幫助的建議。'''
            
    def get_prompt(self, message_list) -> str:
        start_time = time.time()
        try:
            # Use more robust error handling
            response = openai.chat.completions.create(
                model=self.model, 
                messages=message_list
            )

            # Extract content safely
            if response.choices and len(response.choices) > 0:
                content = str(response.choices[0].message.content)

                # Log response details
                end_time = time.time()
                response_time = end_time - start_time
                logging.info(f"Model: {response.model}")
                logging.info(f"Response Time: {response_time:.2f} seconds")
                logging.info(f"Prompt Tokens: {response.usage.prompt_tokens}")
                logging.info(f"Completion Tokens: {response.usage.completion_tokens}")

                return content
            else:
                logging.error("No valid response from the model")
                raise ValueError("No valid response from the model")

        except openai.APIConnectionError as e:
            logging.error(f"OpenAI API Connection Error: {e}")
            raise
        except openai.RateLimitError as e:
            logging.error(f"Rate Limit Exceeded: {e}")
            raise
        except openai.APIError as e:
            logging.error(f"OpenAI API Error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
            raise

    def get_text(self, 
        prompt_words: str,
        prompt_phones : str, 
        predict_phones : str) -> str:
        
        current_message_list = self.message_list.copy()
        text = re.sub("<PROMPT_WORDS>", prompt_words, self.request_template)
        text = re.sub("<PROMPT_PHONES>", prompt_phones, text)
        text = re.sub("<PREDICT_PHONES>", predict_phones, text)
        
        current_message_list.append({"role": "user", "content": text})
        result = ''
        
        try:
            result = self.get_prompt(current_message_list)
        except:
            logging.error('Interaction with chatgpt went error!!')
            return None
        
        return result

if __name__ == "__main__":
    gpt_model = GenerateText()
    result = gpt_model.get_text(prompt_words="we call it bear", prompt_phones="w iy k ao ih t b eh ah", predict_phones="w iy d b ih er")
    print(result)
