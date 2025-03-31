
from openai import OpenAI
import time
import pandas as pd

def get_response(client,model_name,query,retry_num=3,temperature=0.5):

    for _ in range(retry_num):  # Retry
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                        {"role": "system", "content":"You are a helpful assistant."},
                        {"role": "user", "content": query},
                    ],
                temperature = temperature,
                # response_format={"type": "json_object"},
            )
            res = completion.choices[0].message.content
            # if res[-1]!="}":
            #     return None
            # res = json.loads(completion.choices[0].message.content)
            return res
        except Exception as e:
            print(f"Error processing question: {e}")
            time.sleep(10)
    return None


def get_answer(question,related_titles,template):
    api_key = "***"
    client = OpenAI(api_key=api_key)

    model = "gpt-4o-2024-05-13"


    template = template.replace("<question>",question)
    template = template.replace("<related_titles>","\n".join(related_titles))

    response = get_response(client,model,template)
    
    print(response)


if __name__ == "__main__":
    question = "which fields has reinforcement learning been applied to? you must find 40 fields and attach corresponding title reference"
    title_path = r"papers.xlsx"
    template = """
- Role: Academic Research Assistant and Literature Interpretation Expert
- Background: In the process of academic research, users need to quickly and accurately extract key information from a large number of documents to answer specific questions, but are often limited by the complexity and amount of information in the documents. They hope to directly obtain accurate and important answers through a systematic method, and attach references for further research.
- Goals: Based on the provided list of document titles, directly answer the user's question with concise and relevant information, ensuring the answer includes key points and is supported by references from the given titles.
- Constrains: The answer must be strictly based on the content provided by the document titles, and cannot exceed the information available in the titles. Each piece of information in the answer must be clearly referenced to the corresponding title(s) in the provided list.

Related titles:
<related_titles>

Question: <question>
"""

    titles = pd.read_excel(title_path)
    related_titles = titles[titles["title"].str.contains("learning")]["title"].tolist()
    print("len of related_titles:",len(related_titles))

    res  = get_answer(question,related_titles,template)

    print(res)

