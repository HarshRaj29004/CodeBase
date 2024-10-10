# import openai
# from pdfminer.high_level import extract_text
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import BertTokenizer, BertForQuestionAnswering, BertForSequenceClassification


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'r') as file:
        data = file.read()
    lines = data.splitlines()
    qa_pairs = []
    current_qa = {}

    for line in lines:
        if line.startswith('q:'):
            if current_qa:
                qa_pairs.append(current_qa) 
            current_qa = {'q': line[2:].strip()} 
        elif line.startswith('a:'):
            current_qa['a'] = line[2:].strip()  
    if current_qa:
        qa_pairs.append(current_qa)
    return qa_pairs

def search(query, model, cross_enc, embeddings, paragraphs, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]
    cross_input = [[query, paragraphs[hit["corpus_id"]]] for hit in hits]
    cross_scores = cross_enc.predict(cross_input)

    for idx in range(len(cross_scores)):
        hits[idx]["cross_score"] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)

    results = []
    for hit in hits[:5]:
        paragraph = paragraphs[hit["corpus_id"]]
        answer_part = paragraph.split("A:")[1].strip()
        results.append(answer_part.replace("\n", " "))
    return results



def get_bert_answer(tokenizer, model, question, context):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer.strip()


if __name__ == "__main__":
    path = 'dataset.txt'
    qa_pair = extract_text_from_pdf(path)
    model1 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    paragraphs = [f"Q: {qa['q']} A: {qa['a']}" for qa in qa_pair]
    embeddings = model1.encode(
        paragraphs,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    bert_qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',torch_dtype = torch.float16,low_cpu_mem_usage = True)
    bert_qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_qa_model = bert_qa_model.to(device)
    while True:
        query = input("Enter query: ")
    
        results = search(query, model1, cross_encoder, embeddings, paragraphs, top_k=5)

        context = "\n".join(results)  
    
        response = get_bert_answer(bert_qa_tokenizer, bert_qa_model, query, context)

        print("\n***BERT Response***")
        print(response)
