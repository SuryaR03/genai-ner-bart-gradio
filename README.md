# Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

## AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

## PROBLEM STATEMENT:

1. **Entity Recognition**: Identify and classify named entities (e.g., persons, organizations, locations) from unstructured text using a pre-trained and fine-tuned model.

2. **Model Selection and Fine-Tuning**: Utilize a BART model and fine-tune it on a suitable Named Entity Recognition (NER) dataset to improve its accuracy for the task.

3. **User Interaction and Evaluation**: Build an interactive web interface using Gradio to allow users to input text and view recognized entities, facilitating easy evaluation and testing of the NER model's performance.

## DESIGN STEPS:

## STEP 1: Model Selection and Fine-Tuning
Choose a pre-trained BART model (e.g., `facebook/bart-large`).
 Fine-tune the model on a NER dataset (e.g., CoNLL-03).

## STEP 2:Pipeline Creation
Load the fine-tuned BART model and tokenizer using the `transformers` library.
Create a NER pipeline to extract named entities from input text.

## STEP 3: Gradio Interface Development
Develop a Gradio interface to accept text input and display extracted entities.
Deploy the interface for user interaction.


## PROGRAM:
```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
# Helper function
import requests, json

API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "My name is surya, I'm building DeepLearningAI and I live in Chennai"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)
def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["My name is surya and I live in Chennai", "My name is surya and work at AWS"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Andrew, I'm building DeeplearningAI and I live in California", "My name is Poli, I live in Vienna and work at HuggingFace"])

demo.launch(share=True, server_port=int(os.environ['PORT4']))
```
## OUTPUT:

![img](ss.png)

## RESULT:

A fine-tuned BART model successfully identifies and classifies named entities in text.The NER pipeline efficiently extracts entities such as persons, organizations, and locations.The Gradio interface provides an intuitive platform for users to input text and view the results interactively.overall,the  Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework implemented successfully
