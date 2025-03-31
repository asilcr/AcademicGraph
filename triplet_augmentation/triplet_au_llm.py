import os
from openai import OpenAI
import pandas as pd
import prompt
import time
import tqdm
import json



api_key = "sk-"
client = OpenAI(api_key=api_key)


# 加载数据，返回数据列表
def load_data(data_path):
    data_list=[]
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            jso= json.loads(line)
            data_list.append(jso)
    return data_list


def GPT_Chat(client,messages,max_token=prompt.MAX_TOKEN,temperature=prompt.TEMPERATURE,presence_penalty=prompt.PRESENCE_PENALTY):
    # 生成对话的结果
    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",

        messages=messages,
        max_tokens=max_token,
        temperature=temperature, # 控制生成文本的多样性，值越大生成的内容越随机
        presence_penalty=presence_penalty
    )
    return completion.choices[0].message.content


def GPT_NER(ner_type,data,prompt_case,prompt_ask,max_token,temperature):
    ner_res=[]
    ner_data_res=[]
    error_res=[]

    # 批量传入
    if ner_type=="error":
        idx_list=data["idx"].tolist()
        idx_list=[idxs[0] for idxs in idx_list]
    else:
        idx_list=list(range(0,len(data),prompt.NUM_REQUESTS))
    for i in tqdm.tqdm(idx_list):
        text="\n".join([str(x) for x in data[i:i+prompt.NUM_REQUESTS]])

        messages=[
        {"role": "system", "content": prompt_case},
        {"role": "user", "content": prompt_ask+"\n"+text}]
        response=GPT_Chat(client,messages,max_token,temperature)

        try:
            if response[0]=="[":
                response_list=eval(response)
            else:
                response_list=response.split('\n')
            response_list=[x for x in  response_list if x!=""]
            if len(response_list)==prompt.NUM_REQUESTS and "[]" not in str(response):
                for j in range(i,i+prompt.NUM_REQUESTS):
                    ner_data_res.append([data[j]["text"],data[j]["entity"],data[j]["relation"]])
                ner_res.extend(response_list)
            else:
                error_res.append([(i,i+prompt.NUM_REQUESTS),text,response])

        except:
            error_res.append([(i,i+prompt.NUM_REQUESTS),text,response])

        # print("AI_response:",response,"\n----------------------")
        if prompt.Client_is_copilot:
            time.sleep(1)
        # 保存结果
        if (i+prompt.NUM_REQUESTS)%100==0 or i==idx_list[-1]:
            ner_res_df=pd.DataFrame(ner_data_res,columns=["text","pred_entity","pred_relation"])
            ner_res_df['llm_modified']=ner_res
            error_res_df=pd.DataFrame(error_res,columns=["idx","data","gpt_ner"])
            
            if ner_type=="error":
                ner_res_df.to_csv('./data/augmentation_result.csv',index=False)
                error_res_df.to_csv('./data/augmentation_error_result.csv',index=False)
            else:
                ner_res_df.to_csv(prompt.save_path,index=False)
                error_res_df.to_csv(prompt.error_res_path,index=False)

    return ner_res_df,error_res_df




if __name__ == "__main__":
    data = load_data('')
    start_time=time.time()
    ner_type="first"
    gpt_ner_res_df,error_res_df=GPT_NER(ner_type,data,prompt.prompt_case,prompt.prompt_ask,prompt.MAX_TOKEN,prompt.TEMPERATURE,)
    end_time=time.time()
    print("Time cost:{:.4f}s".format(end_time-start_time))
 
    print("Num of modified data:",len(gpt_ner_res_df))
    print("Num of error data:",len(error_res_df)*prompt.NUM_REQUESTS)


