import json
import openai
import os
import logging
import time
#logging.setLevel(logging.WARNING)
class GenerateText:
    proxy = None
    model = ''
    message_list = []
    def __init__(self) -> None:

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


        with open('./config.json', 'r', encoding='utf-8') as inform:
            config = json.load(inform)
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            self.model = config["model"]
            self.message_list.append({"role": "user", "content" : "請以後都使用中文回答我的問題"})
            self.message_list.append({"role": "user", "content" : "參考下列指引作為回答的依據"})
            with open('./pronunciation feedback guideline.txt', 'r', encoding='utf-8') as guideline_file:
                guideline_content = guideline_file.read()
                self.message_list.append({"role": "assistant", "content": guideline_content})
            self.message_list.append({"role": "user", "content" : "有一個八維的發音屬性向量，每一維向量代表口型的位置數據，如第一維度，代表jaw（下巴）的位置，使用數字表示所處的位置：0:Nearly Closed：1:Neutral, 2:Slightly Lowered，如第一維度是0，表示jaw(下巴）近乎關閉，以此類推。第二維度lip separation：0:Closed, 1: Slightly Apart, 2:Apart, 3:Wide Apart。 第三維度lip rounding： 0:Rounded, 1:Slightly Rounded,  2:Neutral,  3:Spread。 第四維度tongue frontness: 0:Back, 1:Slightly Back, 2:Neutral, 3:Slightly Front。第五維度tongue height：0:Low, 1:Mid, 2:Mid-High, 3:High。第六維度tongue tip: 0:low, 1: mid, 2: mid-High 3: High 第七維度velum：0:Closed, 1:Open。第八維度voicing，0:Unvoiced, 1:Voiced。現在需要你依據語音學和生理學的知識，通過計算兩個發音屬性向量（一个為正確的發音屬性向量，一個為實際說話時的發音屬性向量）的相似度，來判斷說話人是否有發音錯誤，如果有請給出原因和建議。例如發IPA phone b時，正確發音屬性向量為（1，0，2，2，1，1，0，1）或（1，2，2，2，1，1，0，1），說話人發音屬性向量為（1，0，2，2，1，1，0，0），要求你通過計算發音向量的相似度，並通過一些語音學和生理學的知識判斷該說話人是否發音錯誤，並給出理由"})
            self.message_list.append({"role": "assistant", "content" : "我們可以通過計算向量的相似度（例如使用歐幾里得距離或餘弦相似度）來量化發音的偏差。對於給出的例子：正確的發音屬性向量為（1，0，2，2，1，1，0，1）或（1，2，2，2，1，1，0，1），說話人的發音屬性向量為（1，0，2，2，1，1，0，0）。我們可以通過比較這兩個向量來判斷發音的準確性。在這個例子中，最後一維（代表發音是否有聲）在正確的發音和說話人的發音中有差異。在正確的發音中，最後一維應為1（有聲），而在說話人的發音中，最後一維為0（無聲）。因此，我們可以得出結論，說話人在發IPA phone b 時發音錯誤，錯誤出在發音的有聲性上。b 應該是一個有聲的輔音，但說話人發出的是無聲的。建議說話人專注於在發出 b 音時確保聲帶振動"})
            self.message_list.append({"role": "user", "content" : "音素中的sil為靜音標籤，不需考慮作為音素的比較，可以將它視作語氣停頓與詞句轉換的起始。"})
            self.message_list.append({"role": "user", "content" : "音素轉發音屬性的關係如下：sh 22133000，iy 01343001 ，hh 22221100，ae 33230001，d  11243401， 12242301，y  12243301，er 22221001，aa 32110001，r  12122301，k  12203100 ，12202100，s  12233300，uw 11013001，t  112434001， 2242300，ih 32342001，n  11223411，g  12203101 ，12202101，w  12003101，ao 32102001，dh 22242201，l  12232401，ow 32132001 ，21012001，m  10221111，eh 32231001，oy 22101001 ，12332001，ay 32220001， 12332001，b  10221101， 12221101，v  20221101，f  20221100，z  12233301，th 22242200，ah 22220001，p  10221100 ，12221100 ，ey 12342001，ng 12203111，ch 22243400， 12133000，uh 12112001，zh 22133001，jh 22243401， 12133001，aw 32220001， 12012001。請你記住這張映射表，每個音素對應8個數字（8維發音屬性向量），或者16個數字（兩個8維發音屬性向量）。問題：跟讀文本：but there came no promise from the bow of the canoe，對應的正確的音素為 b ah t dh eh r k ey m n ow p r aa m ah s f r ah m dh ah b ow ah v dh ah k ah n uw ，實際發音音素為： b ah  dh eh r k eh m n ao p w aa m ah s f r ah n dh ah b ao ah v dh ah k ah n uw ，請你將正確音素和實際音素通過音素轉發音屬性表轉換，轉換完之後，將兩個音素進行分詞，對應到跟讀文本的每個單詞，對比發音屬性的不同，並且你現在需要作為一個英語老師，運用你的教學經驗給出專業的反饋，並且你需要結合每個單詞發音屬性分析的結果，但在給出反饋時候，可以不用給出發音屬性轉換的過程。請給出最主要的錯誤，如替換，刪除，插入這幾類錯誤。"})
            self.message_list.append({"role": "assistant", "content" : "首先，我們將音素轉換為發音屬性向量：正確的發音屬性向量：b: 10221101, ah: 22220001, t: 112434001,dh: 22242201, eh: 32231001, r: 12122301,k: 12203100, ey: 12342001, m: 10221111,n: 11223411, ow: 32132001, p: 10221100,r: 12122301, aa: 32110001, m: 10221111,ah: 22220001, s: 12233300, f: 20221100,r: 12122301, ah: 22220001, m: 10221111,dh: 22242201, ah: 22220001, b: 10221101,ow: 32132001, ah: 22220001, v: 20221101,dh: 22242201, ah: 22220001, k: 12203100,ah: 22220001, n: 11223411, uw: 11013001。實際的發音屬性向量：b: 10221101, ah: 22220001, : n/a,dh: 22242201, eh: 32231001, r: 12122301,k: 12203100, eh: 32231001, m: 10221111,n: 11223411, ao: 32102001, p: 10221100,w: 12003101, aa: 32110001, m: 10221111,ah: 22220001, s: 12233300, f: 20221100,r: 12122301, ah: 22220001, n: 11223411,dh: 22242201, ah: 22220001, b: 10221101,ao: 32102001, ah: 22220001, v: 20221101,dh: 22242201, ah: 22220001, k: 12203100,ah: 22220001, n: 11223411, uw: 11013001。通過對比每個單詞對應的發音屬性向量，如but: 正確發音音素為[b ah t]，實際發音音素為[b ah ]，發音屬性向量完全不匹配，'t'被錯誤地替換為了靜音。最後，我們總結出的錯誤有：替換錯誤：在這個句子的發音中，有多處音素替換錯誤。例如，在 came 的發音中，元音音素 ey 被替換為了 eh。在 promise中，音素 r 被替換為了 w。最後，在 from 中，音素 m 被替換為了 n。為改進這個問題，你需要更多地練習並熟悉每個音素的發音，尤其是元音音素。元音音素在英語中很重要，並且有時候會比較微妙，所以要加以注意。插入錯誤：在這個句子中沒有出現插入錯誤。插入錯誤是指在發音中錯誤地添加了不必要的音素。刪除錯誤：在 but 的發音中，音素 t 被刪除了，變成了靜音。這可能是由於未能清晰地發出詞尾的音。對於這個問題，建議練習清晰發音每個音素，尤其是在詞尾。舌位位置錯誤：這個問題在這個句子中並沒有明顯出現。舌位位置錯誤是指在發音過程中，舌頭的位置放錯了，導致音素發音不準確。如果有舌位位置錯誤的話，可能需要找專業的發音教練或語言治療師來輔導，通過專業的訓練來糾正。"})
            self.message_list.append({"role": "user", "content" : "上面回覆中，反饋再具體並且專業一些，比如音素 m 被替換為了 n，[m] & [n]都是鼻輔音[m]叫做雙唇鼻輔音，發音時雙唇要始終保持緊閉狀態，舌頭放平，自然放鬆，讓氣流從鼻腔中送出；[n]叫做舌尖齒齦鼻輔音，發音時雙唇始終微微張開，舌尖上揚輕輕抵住上齒齦，讓氣流從鼻腔送出，二者的最大區別是嘴巴是否張開，舌頭的擺放位置。而不要回覆“你需要更多地練習並熟悉每個音素的發音，尤其是元音音素。元音音素在英語中很重要，並且有時候會比較微妙，所以要加以注意。”因為這是正確，但是毫無信息量的信息，給不了練習者太多有用信息。請重新回覆剛才的問題，請發揮你作為英語老師的教學經驗，以及從發音屬性對比中得到的信息"})
            self.message_list.append({"role": "assistant", "content" : "了解了你的需求，接下來我將對上述錯誤進行更專業的反饋：替換錯誤：在這個句子的發音中，有多處音素替換錯誤。在 came 的發音中，元音音素 ey 被替換為了 eh。這可能是因為舌頭的位置太低，沒有做到像 ey 那樣中心和前部元音的位置。ey 發音時，口型應略開，舌尖下降，舌後部稍微上升。你可以通過觀察自己的口腔和舌頭的動作，以及聽聽標準的 ey 發音，加強這個音素的練習。在 no和 bow 中，元音 ow 被替換為了 ao。ow 的發音應當從圓唇中元音開始，然後向後降低舌頭位置到低元音。而 ao 的發音舌頭位置一直在後部的位置，這可能是因為你在發 ow 的時候沒有做出這種向後移動的動作。在 promise 中，音素 r 被替換為了 w。這可能是因為你在發 r 的時候舌尖沒有卷起來，這是英語 r 音的特點。最後，在 from 中，音素 m 被替換為了 n。這可能是因為你在發 m 的時候沒有完全閉合嘴唇，導致聲音從口腔而非鼻腔出來。m 音的發音需要嘴唇完全閉合，而 n 音的發音需要舌尖抵住上齒齦。插入錯誤：在這個句子中沒有出現插入錯誤。刪除錯誤：在 but 的發音中，音素 t 被刪除了。這可能是由於你在發 t 的時候舌尖沒有抵住上齒齦，或者沒有用足夠的力氣去爆破。t 音的發音需要舌尖抵住上齒齦，然後用力爆破。我希望這些具體的反饋可以幫助你更好地理解和改善你的發音。請你繼續堅持練習，如果有任何問題，隨時向我詢問。"})        
            self.message_list.append({"role": "user", "content" : "非常好！請以後就按照這個回覆模塊進行回覆。跟讀文本：he looked like one who had passed through an uncomfortable hour or two。正確音素：hh iy l uh k t  l ay k  w ah n hh uw hh ae d  p ae s t th r uw ae n ah n k ah m f er t ah b ah l aw r ao r t uw 。實際音素：hh iy l uh k t z  l ay k  w ah n hh uw hh ae d  p aa s t s r uw ae n ah n k ah m f er t ah b ah ow aw r aa r t uw 。請按照模版進行回覆，不要回覆之前正確，但是沒有信息量，給不了練習者太多有用信息的回覆"})
            self.message_list.append({"role": "assistant", "content" : "了解你的需求，以下是對你提供的實際發音音素與正確音素的對比分析與建議：替換錯誤：passed中的音素 ae 被替換為了 aa。這可能是因為舌頭在發音時的位置偏低。當發 ae 音時，舌頭應位於口腔的前部並稍稍抬高，而 aa 音的舌位較低。你可以多練習 ae 音，觀察口腔形狀和舌位，熟悉這個音素的發音特點。在 or 的發音中，元音音素 ao 被替換為了 aa。這可能是舌尖的位置沒有移到像 ao 那樣的中央，而是像 aa 那樣保持在前面。ao 音需要舌尖稍微後移，而 aa 則舌尖在前。建議多加練習，注意舌尖的位置。插入錯誤：在 looked 和 passed 中，音素 z 和 s 被錯誤地插入了。這可能是你在發 k 和 t 的時候沒有清晰地爆破，而產生了摩擦音。記住 k 和 t 都是爆破音，需要舌尖或舌後抵住上顎，然後用力爆破。刪除錯誤：在 through的發音中，音素 th 被刪除了。這可能是因為你在發 th 的時候舌尖沒有伸到牙齒之間，或者沒有足夠的呼氣形成這個摩擦音。記住 th 音需要舌尖位於上下牙齒之間，並向外稍微吹氣。"})   
            
    # def getPrompt(self, mCczuoUJopenlvmZ6VVQkzo09v7Z3i5GBgkpFF3QI3oiDwessage_list) -> str:
    #     completion = openai.chat.completions.create(
    #     model=self.model, 
    #     messages=message_list
    #     )
    #     print(completion)
    #     return completion['choices'][0]['message']['content']
    
    def getPrompt(self, message_list) -> str:
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

    def getText(self, text : str) -> str:
        current_message_list = self.message_list.copy()
        current_message_list.append({"role": "user", "content": text})
        result = ''
        try:
            result = self.getPrompt(current_message_list)
        except:
            logging.error('interaction with chatgpt error!!')
            return None
        #self.message_list.append({"role": "assistant", "content" : result})
        return result
if __name__ == '__main__':
#    gt = GenerateText()
        #line1 represents transcription text, line2 represents correct phonemes,line3 represents canonical phonemes.
   with open('./data/test/wrd_text', 'r') as file1, \
        open('./data/test/transcript_phn_text', 'r') as file2, \
        open('./data/test/phn_text', 'r') as file3:

    for id, (line1, line2, line3) in enumerate(zip(file1, file2, file3)):

        if id!=4:
            line1 = line1.strip().split(' ',1)
            line2 = line2.strip().split(' ',1)
            line3 = line3.strip().split(' ',1)

            text = (f'''
為我輸出對L2學習者的反饋，
跟讀文本:{line1[1]}。
對應的正確發音pronunciation為：{line2[1]}。
實際的發音pronunciation為: {line3[1]}。
請先對正確pronunciation和實際pronunciation進行分詞，
然後找到每個word對應的pronunciation，
接著通過對比每個word的pronunciation，
找到其中存在的替換，刪除，插入錯誤，
然後對錯誤地方，給出有實際幫助的建議。''')
                        
            print('prompt:')
            print(text)
            print('------------------------------------\n')

            # result = gt.getText(text)
            # print(result)
        else:
            break
