import io
import os
import re
import nltk
import spacy
import docx2txt
from . import constants as cs
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from spacy.matcher import PhraseMatcher
from spacy.lang.en import English
from spacy.lang.en import English
# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor




outdir = './Dataset'
if not os.path.exists(outdir):
    os.mkdir(outdir)

nlp = spacy.load("en_core_web_sm")
skills_patterns_path =os.path.join(outdir, 'Resources/data/skills.jsonl') 
majors_patterns_path = os.path.join(outdir, 'Resources/data/majors.jsonl') 
degrees_patterns_path = os.path.join(outdir, 'Resources/data/degrees.jsonl') 

def extract_text_from_pdf(pdf_path):
    '''
    Helper function to extract the plain text from .pdf files

    :param pdf_path: path to PDF file to be extracted
    :return: iterator of string of extracted text
    '''
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle, codec='utf-8', laparams=LAParams())
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
 
            text = fake_file_handle.getvalue()
            yield text
 
            # close open handles
            converter.close()
            fake_file_handle.close()

def extract_text_from_doc(doc_path):
    '''
    Helper function to extract plain text from .doc or .docx files

    :param doc_path: path to .doc or .docx file to be extracted
    :return: string of extracted text
    '''
    temp = docx2txt.process(doc_path)
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)

def extract_text_from_file(file_path):
    '''
    Wrapper function to detect the file extension and call text extraction function accordingly

    :param file_path: path of file of which text is to be extracted
    :param extension: extension of file `file_name`
    '''
    file_extension = os.path.splitext(file_path)[1].lower()
    text = ''
    if file_extension == '.pdf':
        for page in extract_text_from_pdf(file_path):
            text += ' ' + page
    elif file_extension == '.docx' or file_extension == '.doc':
        text = extract_text_from_doc(file_path)
    return text


def extract_email(text):
    '''
    Helper function to extract email id from text

    :param text: plain text extracted from resume file
    '''
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None
        

def extract_name(nlp_text, matcher):
    '''
    Helper function to extract name from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param matcher: object of `spacy.matcher.Matcher`
    :return: string of full name
    '''
    pattern = [cs.NAME_PATTERN]
    
    matcher.add('NAME', pattern)
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text

def extract_mobile_number(text):
    '''
    Helper function to extract mobile number from text
    :param text: plain text extracted from resume file
    :return: string of extracted mobile numbers
    '''
    phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
    if phone:
        number = ''.join(phone[0])
        if len(number) > 10:
            return '+' + number
        else:
            return number
        
def extract_experience(resume_text):
    '''
    Helper function to extract experience from resume text
    :param resume_text: Plain resume text
    :return: list of experience
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # word tokenization 
    word_tokens = nltk.word_tokenize(resume_text)

    # remove stop words and lemmatize  
    filtered_sentence = [w for w in word_tokens if not w in stop_words and wordnet_lemmatizer.lemmatize(w) not in stop_words] 
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)
    
    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)
    test = []
    
    for vp in list(cs.subtrees(filter=lambda x: x.label()=='P')):
        test.append(" ".join([i[0] for i in vp.leaves() if len(vp.leaves()) >= 2]))
    # Search the word 'experience' in the chunk and then print out the text after it
    x = [x[x.lower().index('experience') + 10:] for i, x in enumerate(test) if x and 'experience' in x.lower()]
    return x


def extract_majors(job):
        nlp = English()
        # Add the pattern to the matcher
        patterns_path = majors_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        acceptable_majors = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'MAJOR':
                if labels_parts[2].replace('-', ' ') not in acceptable_majors:
                    acceptable_majors.append(labels_parts[2].replace('-', ' '))
                if labels_parts[2].replace('-', ' ') not in acceptable_majors:
                    acceptable_majors.append(labels_parts[2].replace('-', ' '))
        return acceptable_majors

def extract_degrees(job):
        nlp = English()
        # Add the pattern to the matcher
        patterns_path = degrees_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        degree_levels = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'DEGREE':
                if labels_parts[1] not in degree_levels:
                    degree_levels.append(labels_parts[1])
        return degree_levels

def extract_skills(job):
        nlp = English()
        patterns_path = skills_patterns_path
        ruler = nlp.add_pipe("entity_ruler")
        ruler.from_disk(patterns_path)
        # Process some text
        doc1 = nlp(job)
        job_skills = []
        for ent in doc1.ents:
            labels_parts = ent.label_.split('|')
            if labels_parts[0] == 'SKILL':
                if labels_parts[1].replace('-', ' ') not in job_skills:
                    job_skills.append(labels_parts[1].replace('-', ' '))
        return ','.join(str(x) for x in job_skills)

def extract_designation(text):
        file = os.path.join(outdir,"Resources/data/titles_combined.txt")
        file = open(file, "r", encoding='utf-8')
        designation = [line.strip().lower() for line in file]
        designitionmatcher = PhraseMatcher(nlp.vocab)
        patterns = [nlp.make_doc(text) for text in designation if len(nlp.make_doc(text)) < 10]
        designitionmatcher.add("Job title", patterns)
        job_titles = []
        __nlp = nlp(text.lower())
        matches = designitionmatcher(__nlp)
        for match_id, start, end in matches:
            span = __nlp[start:end]
            job_titles.append(span.text)
        job_titles = list(set(job_titles))
        return job_titles

def get_full_match(annotations):
    skills=set()
    full_match=annotations['results']['full_matches']
    for dict_item in full_match:
        skills.add(dict_item['doc_node_value'])
    return skills

def get_sub_matches(annotations):
    skills=set()
    sub_match=annotations['results']['ngram_scored']
    for dict_item in sub_match:
        if dict_item['score'] >=0.6:
            skills.add(dict_item['doc_node_value'])  
    return skills

def extract_skills_by_extractor(des):
    try:
        skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
        annotations = skill_extractor.annotate(des)
        full_match=get_full_match(annotations)
        print("Full Match:",len(full_match))
        sub_match=get_sub_matches(annotations)
        print("Sub Match:",len(sub_match))
        full_match=full_match.union(sub_match)
        return ','.join(str(x) for x in full_match)
    except:
        return " "
