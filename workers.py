from PyPDF2 import PdfReader
from question_generation_main import QuestionGeneration
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import spacy
# from gensim.summarization import summarize


# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")


def pdf2text(file_path: str, file_exten: str) -> str:
    """ Converts a given file to text content """

    _content = ''

    # Identify file type and get its contents
    if file_exten == 'pdf':
        with open(file_path, 'rb') as pdf_file:
            _pdf_reader = PdfReader(pdf_file)
            for p in range(len(_pdf_reader.pages)):
                _content += _pdf_reader.pages[p].extract_text()
            # _content = _pdf_reader.getPage(0).extractText()
            print('PDF operation done!')

    elif file_exten == 'txt':
        with open(file_path, 'r') as txt_file:
            _content = txt_file.read()
            print('TXT operation done!')

    return _content



def txt2questions(doc: str, n=10, o=4) -> dict:
    """ Get all questions and options """

    qGen = QuestionGeneration(n, o)
    q = qGen.generate_questions_dict(doc)
    for i in range(len(q)):
        temp = []
        for j in range(len(q[i + 1]['options'])):
            temp.append(q[i + 1]['options'][j + 1])
        # print(temp)
        q[i + 1]['options'] = temp
    return q



def extractive_summarize(text, num_sentences):
    # Tokenize the text into sentences using a regular expression
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    # If there are too few sentences, return the entire text
    if len(sentences) <= num_sentences:
        return text

    # Process the text with spaCy to recognize names and entities
    doc = nlp(text)
    
    # Collect recognized names and entities
    named_entities = set()
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:  # Customize the entity types as needed
            named_entities.add(ent.text)

    # Vectorize the sentences
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Calculate sentence similarity using cosine similarity
    similarity_matrix = cosine_similarity(vectors)

    # Rank sentences based on similarity and named entity presence
    sentence_scores = np.zeros(len(sentences))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sentence_scores[i] += similarity_matrix[i][j]
                for entity in named_entities:
                    if entity in sentences[i]:
                        sentence_scores[i] += 1  # Add weight for sentences containing named entities

    # Get the top-ranked sentences for the summary, preserving their original order
    ranked_sentences = [(sentence_scores[i], sentence, i) for i, sentence in enumerate(sentences)]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[0], reverse=True)[:num_sentences]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[2])

    # Construct the summary in the original sequence
    summary = " ".join([ranked_sentence[1] for ranked_sentence in ranked_sentences])

    return summary




def summarize_pdf(file_path, file_exten, num_sentences=10):
    # Convert the PDF to text
    pdf_text = pdf2text(file_path, file_exten)
    
    # Summarize the text using extractive summarization
    summary = extractive_summarize(pdf_text, num_sentences)
    return summary
