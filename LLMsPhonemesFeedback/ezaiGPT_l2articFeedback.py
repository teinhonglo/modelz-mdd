from gpt4dataset＿generate import GenerateText
import pandas as pd
import re
import os
from datetime import datetime


results = []

# 確保輸出目錄存在
os.makedirs('./output', exist_ok=True)

gt = GenerateText()
with open('./data/test/wrd_text', 'r') as file1, \
        open('./data/test/transcript_phn_text', 'r') as file2, \
        open('./data/test/phn_text', 'r') as file3:
    
    for id, (line1, line2, line3) in enumerate(zip(file1, file2, file3)):
        # 處理每一行數據
        line1 = line1.strip().split(' ', 1)
        line2 = line2.strip().split(' ', 1)
        line3 = line3.strip().split(' ', 1)
        
        if len(line1) < 2 or len(line2) < 2 or len(line3) < 2:
            print(f"Skipping line {id}: Invalid format")
            continue
        
        # 創建提示文本
        text = f'''為我輸出對L2學習者的反饋，
跟讀文本:{line1[1]}。
對應的正確發音pronunciation為：{line2[1]}。
實際的發音pronunciation為: {line3[1]}。
請先對正確pronunciation和實際pronunciation進行分詞，
然後找到每個word對應的pronunciation，
接著通過對比每個word的pronunciation，
找到其中存在的替換，刪除，插入錯誤，
然後對錯誤地方，給出有實際幫助的建議。'''

        # 獲取反饋
        result = gt.getText(text)

        results.append({
            'ID': id,
            # 'Reading_Text': line1[1],
            # 'Correct_Pronunciation': line2[1],
            # 'Actual_Pronunciation': line3[1],
            gt.model + '_Full_Feedback with phonemes in English': result
        })
        print(f"Processed line {id}")

        if id==3:break

file_path = './output/gpt_series_pronunciation_feedback.csv'
results = pd.DataFrame(results)
df = pd.read_csv(file_path, encoding='utf-8')
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df = pd.merge(df, results, on='ID', how='outer')
df.to_csv(file_path, index=False, encoding='utf-8')
print(f"\nResults saved to {file_path}")