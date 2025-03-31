Triplet_augmentation_prompt="""
You are an expert in artificial intelligence, please correct the joint entity and relationships extraction results.

Entity definition:
"Problem": research topic, purpose and main work, mostly in nouns and gerunds.
"Method": algorithms, models and theories used to solve problems or achieve research goals, often paired with "using/via/with/based on/by'.
"Tool": software, resources or channels used to implement research experiments, or the general term for a certain solution, which is often paired with "using/via/with/based on/by'.
"Field":  research industry, often paired with "in/within' or before "Problem" entities.
"Perspective": specific subdivision direction and characteristics of research, often as an adjective.
"Feature": characteristics of research methods and tools, often as adjectives or paired with 'based'.
"Application": specific application scenario of research content.
"Condition": research prerequisites and special requirements, often paired with 'with/without'.
"Dataset": data used in the task.

Relationship definition:
- "subclass of": (Problem, Problem),(Method, Method)
- "parallel to": (Problem, Problem),(Method, Method),(Tool, Tool)
- "belong to": (Perspective, Problem)
- "restrict to": (Condition, Method),(Condition, Problem),(Condition, Tool)
- "lie in": (Problem, Field),(Method, Field)
- "serve for": (Dataset, Problem),(Dataset, Tool),(Dataset, Method)
- "solved by": (Problem, Method),(Problem, Tool)
- "feature of" (Feature, Method),(Feature, Tool)
- "applied in": (Method, Application),(Problem, Application),(Tool, Application)
- "used in": (Tool, Method)

Notes:
1. Extracted entities must be short and concise, about 2-5 words in length, and can not be repeated.
2. Pay special attention to that prepositions or conjunctions can not be appeared in the beginning or end of entities, such as "for/of/and/to/a/the/on/'/in", which should be removed.
3. Feature and perspective need to be extracted separately from task, method and tool.
4. Remove "based" or "-based" at the end of enetity.
5. When the entity is modified, the corresponding relationship must be improved.

Outputs:
The output should be in json format, structured as follows:
{"entities":{},
"relations":{}}

Case1:
{"text": "Improving data augmentation for low resource speech-to-text translation with diverse paraphrasing", "entity": {"Tool": ["data augmentation"], "Problem": ["speech-to-text translation"]}, "relation": {}}
outputs: {"entities": {"Method": ["data augmentation"], "Problem": ["speech-to-text translation"], "Perspective": ["low resource"], "Condition": ["with diverse paraphrasing"]}, "relation": {"solved by":[["speech-to-text translation", "data augmentation"]], "restrict to":[["low resource", "data augmentation"], ["with diverse paraphrasing", "data augmentation"]]}}

case2:
{"text": "Adversarial attacks on multi-focus image fusion models", "entity": {"Problem": ["Adversarial attacks"], "Tool": ["image fusion"]}, "relation": {}}
outputs:{"entities": {"Problem": ["Adversarial attacks"], "Method": ["image fusion model"],"Feature": ["multi-focus"]}, "relation": {"Feature of":[["multi-focus", "image fusion model"]], "solved by":[["Adversarial attacks", "image fusion model}}

Please strictly follow the above notes and cases to extract the predefined entity and their relationships from following sentence:
{sentence} # for triplet extraction from scratch

output:
"""