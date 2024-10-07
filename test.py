from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from text_generation import Client
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PREPROMPT = "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful.\n"
PROMPT = """"Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to
make up an answer. Don't make up new terms which are not available in the context.

{context}"""

END_7B = "\n<|prompter|>{query}<|endoftext|><|assistant|>"
END_40B = "\nUser: {query}\nFalcon:"

PARAMETERS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "top_k": 50,
    "truncate": 1000,
    "max_new_tokens": 1024,
    "seed": 42,
    "stop_sequences": ["<|endoftext|>", "</s>"],
}
CLIENT_7B = Client("http://127.0.0.1.3000")  # Fill this part
# CLIENT_40B = Client("https://")  # Fill this part

def extract_text_from_pdf(pdf_path):
    raw_text = extract_text(pdf_path)
    cleaned_text = " ".join(raw_text.split()) 
    return cleaned_text

def process_extracted_text_to_qa(extracted_text):
    qa_pairs = []
    blocks = extracted_text.split("q:")[1:] 
    for block in blocks:
        try:
            question, answer = block.split("a:", 1)
            question = question.strip()
            answer = answer.strip()
            qa_pairs.append({"q": question, "a": answer})
        except ValueError:
            continue
    return qa_pairs

def embed_qa_pairs(qa_pairs):
    paragraphs = [f"Q: {qa['q']} A: {qa['a']}" for qa in qa_pairs]
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    embeddings = model.encode(
        paragraphs,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return model, embeddings, paragraphs, cross_encoder

def search(query, model, cross_enc, embeddings, paragraphs, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]
    cross_input = [[query, paragraphs[hit["corpus_id"]]] for hit in hits]
    cross_scores = cross_enc.predict(cross_input)

    for idx in range(len(cross_scores)):
        hits[idx]["cross_score"] = cross_scores[idx]

    results = []
    hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)
    for hit in hits[:5]:
        results.append(paragraphs[hit["corpus_id"]].replace("\n", " "))
    return results


def load_falcon_model():
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    model = AutoModelForCausalLM.from_pretrained("distilbert-base-uncased", torch_dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


def generate_falcon_response(model, tokenizer, prompt, max_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=max_tokens, 
        temperature=0.7, 
        top_p=0.95, 
        top_k=50, 
        no_repeat_ngram_size=2,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    path = 'Dataset.pdf'
    text = extract_text_from_pdf(path)
    qa_pair = process_extracted_text_to_qa(text)
    model1, embedding, para, cross_enc = embed_qa_pairs(qa_pair)
    model, tokenizer = load_falcon_model()
    while True:
        query = input("Enter query: ")
        results = search(query, model1, cross_enc, embedding, para, top_k=5)

        query_7b = PREPROMPT + PROMPT.format(context="\n".join(results))
        query_7b += END_7B.format(query=query)

    
        falcon_response = generate_falcon_response(model, tokenizer, query_7b)

 
        print("\n***Falcon Response***")
        print(falcon_response)
