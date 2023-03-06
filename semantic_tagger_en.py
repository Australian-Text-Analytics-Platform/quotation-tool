#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:29:44 2022

@author: sjuf9909
"""
# import required packages
import codecs
import hashlib
import io
import os
import sys
from tqdm import tqdm
import zipfile
from zipfile import ZipFile
from pyexcelerate import Workbook
from collections import Counter
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")
import joblib
import itertools

# numpy and pandas: tools for data processing
import pandas as pd
import numpy as np

# matplotlib: visualization tool
from matplotlib import pyplot as plt
#from matplotlib import font_manager

# spaCy and NLTK: natural language processing tools for working with language/text data
import spacy
#from spacy.tokens import Doc
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# ipywidgets: tools for interactive browser controls in Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display, clear_output, FileLink, HTML

class DownloadFileLink(FileLink):
    '''
    Create link to download files in Jupyter Notebook
    '''
    html_link_str = "<a href='{link}' download={file_name}>{link_text}</a>"

    def __init__(self, path, file_name=None, link_text=None, *args, **kwargs):
        super(DownloadFileLink, self).__init__(path, *args, **kwargs)

        self.file_name = file_name or os.path.split(path)[1]
        self.link_text = link_text or self.file_name

    def _format_path(self):
        from html import escape

        fp = "".join([self.url_prefix, escape(self.path)])
        return "".join(
            [
                self.result_html_prefix,
                self.html_link_str.format(
                    link=fp, file_name=self.file_name, link_text=self.link_text
                ),
                self.result_html_suffix,
            ]
        )
        

class SemanticTagger():
    '''
    Rule based token and Multi Word Expression semantic tagger for the English language
    '''
    
    def __init__(self):
        '''
        Initiate the SemanticTagger
        '''
        # initiate other necessary variables
        self.mwe_count = 0
        self.text_df = None
        self.tagged_df = None
        self.large_file_size=1000000
        self.token_to_display=500
        self.max_to_process = 1000
        self.cpu_count = joblib.cpu_count()
        self.selected_text = {'left': None, 'right': None}
        self.text_name = {'left': None, 'right': None}
        self.df = {'left': None, 'right': None}
        
        # create an output folder if not already exist
        os.makedirs('output', exist_ok=True)
        
        # get usas_tags definition
        usas_def_file = './documents/semtags_subcategories.txt'
        self.usas_tags = dict()
        
        # reading line by line
        with open(usas_def_file) as file_x:
            for line in file_x:
                self.usas_tags[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]
        
        # initiate the variables for file uploading
        self.file_uploader = widgets.FileUpload(
            description='Upload your files (txt, csv, xlsx or zip)',
            accept='.txt, .xlsx, .csv, .zip', # accepted file extension
            multiple=True,  # True to accept multiple files
            error='File upload unsuccessful. Please try again!',
            layout = widgets.Layout(width='320px')
            )
    
        self.upload_out = widgets.Output()
        
        # give notification when file is uploaded
        def _cb(change):
            with self.upload_out:
                if self.file_uploader.value!=():
                    # clear output and give notification that file is being uploaded
                    clear_output()
                    
                    # check file size
                    self.check_file_size(self.file_uploader)
                    
                    # reading uploaded files
                    self.process_upload()
                    
                    # give notification when uploading is finished
                    print('Finished uploading files.')
                    print('{} text documents are loaded for tagging.'.format(self.text_df.shape[0]))
                
                # clear saved value in cache and reset counter
                self.file_uploader.value = ()
            
        # observe when file is uploaded and display output
        self.file_uploader.observe(_cb, names='value')
        self.upload_box = widgets.VBox([self.file_uploader, self.upload_out])
        
        # CSS styling 
        self.style = """
        <style scoped>
            .dataframe-div {
              max-height: 350px;
              max-width: 1050px;
              overflow: auto;
              position: relative;
            }
        
            .dataframe thead th {
              position: -webkit-sticky; /* for Safari */
              position: sticky;
              top: 0;
              background: #2ca25f;
              color: white;
            }
        
            .dataframe thead th:first-child {
              left: 0;
              z-index: 1;
            }
        
            .dataframe tbody tr th:only-of-type {
                    vertical-align: middle;
                }
        
            .dataframe tbody tr th {
              position: -webkit-sticky; /* for Safari */
              position: sticky;
              left: 0;
              background: #99d8c9;
              color: white;
              vertical-align: top;
            }
        </style>
        """
        
        
    def loading_semantic_tagger(self, 
                                language: str, 
                                mwe: str):
        '''
        loading spaCy language model and PyMUSAS pipeline based on the selected language

        Args:
            language: the language selected by user
            mwe: whether to include Multi-Word Expressions (MWE) detection
        '''
        # the different parameters for different languages
        languages = {'english':
                     {'yes':{'spacy_lang_model':'en_core_web_sm',
                             'exclude':['ner', 'parser', 'entity_linker', 'entity_ruler', 'morphologizer', 'transformer'],
                             'pymusas_tagger':'en_dual_none_contextual'},
                      'no':{'spacy_lang_model':'en_core_web_sm',
                            'exclude':['ner', 'parser', 'entity_linker', 'entity_ruler', 'morphologizer', 'transformer'],
                            'pymusas_tagger':'en_single_none_contextual'}}
                     }
        
        if self.mwe_count==0:
            # give warning loading the pipeline may take a while
            print('Loading the Semantic Tagger for the English language...')
            print('This may take a while...')
            
            # download spaCy's language model for the selected language
            # and exclude unnecessary components
            self.nlp = spacy.load(languages[language][mwe]['spacy_lang_model'], 
                                  exclude=languages[language][mwe]['exclude'])
            
            # load the PyMUSAS rule based tagger in a separate spaCy pipeline
            tagger_pipeline = spacy.load(languages[language][mwe]['pymusas_tagger'])
            
            # adds the PyMUSAS rule based tagger to the main spaCy pipeline
            self.nlp.add_pipe('pymusas_rule_based_tagger', source=tagger_pipeline)
            self.nlp.add_pipe('sentencizer')
            print('Finished loading.')
            warnings.filterwarnings("default")
            self.mwe_count+=1
        else:
            print('\nSemantic tagger has been loaded and ready for use.')
            print('Please re-start the kernel if you wish to select a different option (at the top, select Kernel - Restart).')
    
    
    def loading_tagger_widget(self):
        '''
        Create a widget to select the semantic tagger language and the mwe option
        '''
        # widget to display instruction
        enter_language = widgets.HTML(
            value='<b>Select a language:</b>',
            placeholder='',
            description=''
            )
        
        # select language
        select_language = widgets.Dropdown(
            options=['english'],
            value='english',
            description='',
            disabled=False,
            layout = widgets.Layout(width='200px')
        )
        
        # widget to display instruction
        enter_text = widgets.HTML(
            value='Would you like to include multi-word expressions (mwe)?',
            placeholder='',
            description=''
            )
        
        enter_text2 = widgets.HTML(
            value = '<b><font size=2.5>Warning:</b> including mwe extraction will make the process much slower. \
                For a corpus >500 texts, we recommend choosing the non-MWE version.</b>',
            #value='<b>Warning:</b> including mwe extraction will make the process much slower.',
            placeholder='',
            description=''
            )
        
        # select mwe
        mwe_selection = widgets.RadioButtons(
            options=['yes', 'no'],
            value='yes', 
            layout={'width': 'max-content'}, # If the items' names are long
            description='',
            disabled=False
            )
        
        # widget to show loading button
        load_button, load_out = self.click_button_widget(desc='Load Semantic Tagger', 
                                                       margin='15px 0px 0px 0px',
                                                       width='180px')
        
        # function to define what happens when the button is clicked
        def on_load_button_clicked(_):
            with load_out:
                clear_output()
                
                language = select_language.value
                self.mwe = mwe_selection.value
                
                # load selected language semantic tagger
                self.loading_semantic_tagger(language, self.mwe)
                enter_language.value = 'Semantic tagger for {} language has been loaded'.format(language)
                select_language.options = [language]
                
                if self.mwe=='no':
                    enter_text.value = 'Semantic tagger without MWE extraction has been loaded and is ready for use.'
                    enter_text2.value=''
                    mwe_selection.options=['no']
                else:
                    enter_text.value = 'Semantic tagger with MWE extraction has been loaded and is ready for use.'
                    mwe_selection.options=['yes']
                
        # link the button with the function
        load_button.on_click(on_load_button_clicked)
        
        vbox = widgets.VBox([enter_text, 
                             mwe_selection, 
                             enter_text2, 
                             load_button, load_out])
        
        return vbox

    
    def click_button_widget(
            self, 
            desc: str, 
            margin: str='10px 0px 0px 10px',
            width='320px'
            ):
        '''
        Create a widget to show the button to click
        
        Args:
            desc: description to display on the button widget
            margin: top, right, bottom and left margins for the button widget
        '''
        # widget to show the button to click
        button = widgets.Button(description=desc, 
                                layout=Layout(margin=margin, width=width),
                                style=dict(font_style='italic',
                                           font_weight='bold'))
        
        # the output after clicking the button
        out = widgets.Output()
        
        return button, out
    
    
    def check_file_size(self, uploaded_file):
        '''
        Function to check the uploaded file size
        
        Args:
            uploaded_file: the uploaded file containing the text data
        '''
        # check total uploaded file size
        total_file_size = sum([file['size'] for file in uploaded_file.value])
        print('The total size of the upload is {:.2f} MB.'.format(total_file_size/1000000))
        
        # display warning for individual large files (>1MB)
        large_text = [file['name'] for file in uploaded_file.value \
                      if file['size']>self.large_file_size and \
                          file['name'].endswith('.txt')]
        if len(large_text)>0:
            print('The following file(s) are larger than 1MB:', large_text)
        
        
    def extract_zip(self, zip_file):
        '''
        Load zip file
        
        Args:
            zip_file: the file containing the zipped data
        '''
        # create an input folder if not already exist
        os.makedirs('input', exist_ok=True)
        
        # read and decode the zip file
        temp = io.BytesIO(zip_file['content'])
        
        # open and extract the zip file
        with ZipFile(temp, 'r') as zip:
            # extract files
            print('Extracting {}...'.format(zip_file['name']))
            zip.extractall('./input/')
        
        # clear up temp
        temp = None
    
    
    def load_txt(self, file, n) -> list:
        '''
        Load individual txt file content and return a dictionary object, 
        wrapped in a list so it can be merged with list of pervious file contents.
        
        Args:
            file: the file containing the text data
            n: index of the uploaded file (value='unzip' if the file is extracted form a zip file
        '''
        # read the unzip text file
        if n=='unzip':
            # read the unzip text file
            with open(file) as f:
                temp = {'text_name': file.name[:-4],
                        'text': f.read()
                }
            
            os.remove(file)
        else:
            file = self.file_uploader.value[n]
            # read and decode uploaded text
            temp = {'text_name': file['name'][:-4],
                    'text': codecs.decode(file['content'], encoding='utf-8', errors='replace')
            }
            
            # check for unknown characters and display warning if any
            unknown_count = temp['text'].count('�')
            if unknown_count>0:
                print('We identified {} unknown character(s) in the following text: {}'.format(unknown_count, file['name'][:-4]))
        
        return [temp]


    def load_table(self, file, n) -> list:
        '''
        Load csv or xlsx file
        
        Args:
            file: the file containing the excel or csv data
            n: index of the uploaded file (value='unzip' if the file is extracted form a zip file
        '''
        if n!='unzip':
            file = io.BytesIO(self.file_uploader.value[n]['content'])
            
        # read the file based on the file format
        try:
            temp_df = pd.read_csv(file)
        except:
            temp_df = pd.read_excel(file)
            
        # check if the column text and text_name present in the table, if not, skip the current spreadsheet
        if ('text' not in temp_df.columns) or ('text_name' not in temp_df.columns):
            print('File {} does not contain the required header "text" and "text_name"'.format(self.file_uploader.value[n]['name']))
            return []
        
        # return a list of dict objects
        temp = temp_df[['text_name', 'text']].to_dict(orient='index').values()
        
        return temp
    
    
    def hash_gen(self, temp_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create column text_id by md5 hash of the text in text_df
        
        Args:
            temp_df: the temporary pandas dataframe containing the text data
        '''
        temp_df['text_id'] = temp_df['text'].apply(lambda t: hashlib.md5(t.encode('utf-8')).hexdigest())
        
        return temp_df
    
    
    def process_upload(self, deduplication: bool = True):
        '''
        Pre-process uploaded .txt files into pandas dataframe

        Args:
            deduplication: option to deduplicate text_df by text_id
        '''
        # create placeholders to store all texts and zipped file names
        all_data = []; files = []
        
        # read and store the uploaded files
        uploaded_files = self.file_uploader.value
        
        # extract zip files (if any)
        for n, file in enumerate(uploaded_files):
            files.append([file.name, n])
            if file.name.lower().endswith('zip'):
                self.extract_zip(self.file_uploader.value[n])
                files.pop()
        
        # add extracted files to files
        for file_type in ['*.txt', '*.xlsx', '*.csv']:
            files += [[file, 'unzip'] for file in Path('./input').rglob(file_type) if 'MACOSX' not in str(file)]
        
        print('Reading uploaded files...')
        print('This may take a while...')
        # process and upload files
        for file, n in tqdm(files):
            # process text files
            if str(file).lower().endswith('txt'):
                text_dic = self.load_txt(file, n)
            # process xlsx or csv files
            else:
                text_dic = self.load_table(file, n)
            all_data.extend(text_dic)
        
        # remove files and directory once finished
        os.system('rm -r ./input')
        
        # convert them into a pandas dataframe format and add unique id
        self.text_df = pd.DataFrame.from_dict(all_data)
        self.text_df = self.hash_gen(self.text_df)
        
        # clear up all_data
        all_data = []; files = []
        
        # deduplicate the text_df by text_id
        if deduplication:
            self.text_df.drop_duplicates(subset='text_id', keep='first', inplace=True)
        
        def clean_text(text):
            '''
            Function to clean the text

            Args:
                text: the text to be cleaned
            '''
            # clean empty spaces in the text
            text = re.sub(r'\n','', text)
            
            return text
        
        print('Pre-processing uploaded text...')
        tqdm.pandas()
        self.text_df['text'] = self.text_df['text'].progress_apply(clean_text)
        
    
    def check_mwe(self, token) -> str:
        '''
        Function to check if a token is part of multi-word expressions

        Args:
            token: the spaCy token to check
        '''
        return ['yes' if (token._.pymusas_mwe_indexes[0][1]-\
                         token._.pymusas_mwe_indexes[0][0])>1 else 'no'][0]
        
    
    def remove_symbols(self, text: str) -> str:
        '''
        Function to remove special symbols from USAS tags

        Args:
            text: the USAS tag to check
        '''
        text = re.sub('m','',text)
        text = re.sub('f','',text)
        text = re.sub('%','',text)
        text = re.sub('@','',text)
        text = re.sub('c','',text)
        text = re.sub('n','',text)
        text = re.sub('i','',text)
        text = re.sub(r'([+])\1+', r'\1', text)
        text = re.sub(r'([-])\1+', r'\1', text)
        
        return text
    
    
    def usas_tags_def(self, token) -> list:
        '''
        Function to interpret the definition of the USAS tag

        Args:
            token: the token containing the USAS tag to interpret
        '''
        try: 
            usas_tags = token._.pymusas_tags[0].split('/')
            if usas_tags[-1]=='':
                usas_tags = usas_tags[:-1]
        except: 
            usas_tags = 'Z99'.split('/')
        
        tag_def = []
        tags = []
        for usas_tag in usas_tags:
            tag = self.remove_symbols(usas_tag)
            if tag=='PUNCT':
                tag_def.append(usas_tag)
            else:
                while tag not in self.usas_tags.keys() and tag!='':
                    tag=tag[:-1]
                try: tag_def.append(self.usas_tags[tag])
                except: tag_def.append(usas_tag)
            tags.append(tag)
            
        return tags, tag_def
    
    
    def token_usas_tags(self, token) -> str:
        '''
        Function to add the USAS tag to the token

        Args:
            token: the token to be added with the USAS tag 
        '''
        try:
            token_tag = token.text + ' (' + self.usas_tags_def(token)[0] + ', ' + self.usas_tags_def(token)[1] + ')'
        except:
            token_tag = token.text + ' (' + self.usas_tags_def(token)[0] + ')'
        
        return token_tag
    
    
    def highlight_sentence(self, token) -> str:
        '''
        Function to highlight selected token in the sentence

        Args:
            token: the token to be highlighted
        '''
        sentence = token.sent
        start_index = token._.pymusas_mwe_indexes[0][0]
        end_index = token._.pymusas_mwe_indexes[0][1]
        
        # highlight multi-words for MWE
        if end_index-start_index>1:
            new_sentence = []
            for token in sentence:
                if token.i==start_index:
                    highlight_word_start = '<span style="color: #2ca25f; font-weight: bold">{}'.format(str(token.text))
                    new_sentence.append(highlight_word_start+token.whitespace_)
                elif token.i==end_index-1:
                    highlight_word_end = '{}</span>'.format(str(token.text))
                    new_sentence.append(highlight_word_end+token.whitespace_)
                else:
                    new_sentence.append(token.text+token.whitespace_)
            text = ''.join(new_sentence)
        # for non-MWE, just highlight the token
        else:
            word = token.text
            word_index = token.i
            highlight_word = '<span style="color: #2ca25f; font-weight: bold">{}</span>'.format(word)
            text = ''.join([token.text+token.whitespace_ \
                            if token.i!=word_index else highlight_word+token.whitespace_ \
                                for token in sentence])
        
        return text
    
    
    def add_tagger(self, 
                   text_name: str,
                   text_id:str, 
                   doc) -> pd.DataFrame:
        '''
        add semantic tags to the texts and convert into pandas dataframe

        Args:
            text_name: the text_name of the text to be tagged by the semantic tagger
            text_id: the text_id of the text to be tagged by the semantic tagger
            text: the text to be tagged by the semantic tagger
        '''
        if self.mwe=='yes':
            # extract the semantic tag for each token
            tagged_text = [{'text_name':text_name,
                            'text_id':text_id,
                            'token':token.text,
                            'pos':token.pos_,
                            'usas_tags': self.usas_tags_def(token)[0],
                            'usas_tags_def': self.usas_tags_def(token)[1],
                            'mwe': self.check_mwe(token),
                            'lemma':token.lemma_,
                            'sentence':self.highlight_sentence(token),
                            'start_index': token._.pymusas_mwe_indexes[0][0],
                            'end_index': token._.pymusas_mwe_indexes[0][1]} for token in doc]
        else:
            # extract the semantic tag for each token
            tagged_text = [{'text_name':text_name,
                            'text_id':text_id,
                            'token':token.text,
                            'pos':token.pos_,
                            'usas_tags': self.usas_tags_def(token)[0], 
                            'usas_tags_def': self.usas_tags_def(token)[1], 
                            'lemma':token.lemma_,
                            'sentence':self.highlight_sentence(token),
                            'start_index': token._.pymusas_mwe_indexes[0][0],
                            'end_index': token._.pymusas_mwe_indexes[0][1]} for token in doc]
        
        # convert output into pandas dataframe
        tagged_text_df = pd.DataFrame.from_dict(tagged_text)
        
        return tagged_text_df
    
    
    def tag_text(self):
        '''
        Function to iterate over uploaded texts and add semantic taggers to them
        '''
        if self.mwe=='no':
            if len(self.text_df)<500:
                n_process=1
            else:
                n_process=self.cpu_count
        else:
            n_process=1
        # iterate over texts and tag them
        for n, doc in enumerate(tqdm(self.nlp.pipe(self.text_df['text'].to_list(),
                                                n_process=n_process),
                                  total=len(self.text_df))):
            try:
                text_name = self.text_df.text_name[self.text_df.index[n]]
                text_id = self.text_df.text_id[self.text_df.index[n]]
                tagged_text = self.add_tagger(text_name, 
                                              text_id, 
                                              doc)
                self.tagged_df = pd.concat([self.tagged_df,tagged_text])
            
            except:
                # provide warning if text is too large
                print('{} is too large. Consider breaking it \
                      down into smaller texts (< 1MB each file).'.format(text_name))
        
        # reset the pandas dataframe index after adding new tagged text
        self.tagged_df.reset_index(drop=True, inplace=True)
        
        
    def display_tag_text(self, left_right: str): 
        '''
        Function to display tagged texts 
        '''
        # widgets for selecting text_name to analyse
        enter_text, text = self.select_text_widget()
        
        # widget to analyse tags
        display_button, display_out = self.click_button_widget(desc='Display tagged text',
                                                       margin='12px 0px 0px 0px',
                                                       width='150px')
        
        # the widget for statistic output
        stat_out = widgets.Output()
        
        # widget to filter pos
        filter_pos, select_pos = self.select_multiple_options('<b>pos:</b>',
                                                              ['all'],
                                                              ['all'])
        
        # widget to filter usas_tags
        filter_usas, select_usas = self.select_multiple_options('<b>usas tag:</b>',
                                                              ['all'],
                                                              ['all'])
        
        # widget to filter mwe
        filter_mwe, select_mwe = self.select_multiple_options('<b>mwe:</b>',
                                                              ['all','yes','no'],
                                                              ['all'])
        
        # function to define what happens when the button is clicked
        def on_display_button_clicked(_):
            # display selected tagged text
            with display_out:
                clear_output()
                
                # get selected text
                self.text_name[left_right] = text.value
                
                # display the selected text
                self.df[left_right] = self.tagged_df[self.tagged_df['text_name']==self.text_name[left_right]].iloc[:,2:].reset_index(drop=True)
                
                # for new selected text
                if self.text_name[left_right]!=self.selected_text[left_right]:
                    self.selected_text[left_right]=self.text_name[left_right]
                    
                    # generate usas tag options
                    usas_list = self.df[left_right].usas_tags_def.to_list()
                    usas_list = [item for sublist in usas_list for item in sublist]
                    usas_list = sorted(list(set(usas_list)))
                    usas_list.insert(0,'all')
                    select_usas.options = usas_list
                    
                    # generate pos options
                    new_pos = sorted(list(set(self.df[left_right].pos.to_list())))
                    new_pos.insert(0,'all')
                    select_pos.options = new_pos
                    
                    select_pos.value=('all',)
                    select_usas.value=('all',)
                    select_mwe.value=('all',)
                
                # get the filter values
                inc_pos = select_pos.value
                inc_usas = select_usas.value
                inc_mwe = select_mwe.value
                
                # display based on selected filter values
                if inc_usas!=('all',):
                    usas_index=[]
                    for selected_usas in inc_usas:
                        index = [n for n, item in enumerate(self.df[left_right].usas_tags_def.to_list()) \
                                 if selected_usas in item]
                        usas_index.extend(index)
                    usas_index = list(set(usas_index))
                    self.df[left_right] = self.df[left_right].iloc[usas_index]
                
                if inc_pos!=('all',):
                    self.df[left_right] = self.df[left_right][self.df[left_right]['pos'].isin(inc_pos)]
                
                if inc_mwe!=('all',):
                    self.df[left_right] = self.df[left_right][self.df[left_right]['mwe'].isin(inc_mwe)]
                
                pd.set_option('display.max_rows', None)
                
                print('Tagged text: {}'.format(self.text_name[left_right]))
                # display in html format for styling purpose
                if inc_usas==('all',) and inc_pos==('all',) and inc_mwe==('all',):
                    # only displays the first n tokens, with n defined by self.token_to_display
                    print('The below table shows the first {} tokens only. Use the above filter to show tokens with specific tags.'.format(self.token_to_display))
                    df_html = self.df[left_right].head(self.token_to_display).iloc[:,:-2].to_html(escape=False)
                else:
                    df_html = self.df[left_right].iloc[:,:-2].to_html(escape=False)
                
                # Concatenating to single string
                df_html = self.style+'<div class="dataframe-div">'+df_html+"\n</div>"
                
                display(HTML(df_html))
                
            with stat_out:
                clear_output()
                print('Statistical information: {}'.format(self.text_name[left_right]))
                count_usas = pd.DataFrame.from_dict(Counter(list(itertools.chain(*self.df[left_right].usas_tags_def.to_list()))),
                                                    orient='index', columns=['usas_tag']).T
                count_pos = pd.DataFrame.from_dict(Counter(self.df[left_right].pos.to_list()),
                                                   orient='index', columns=['pos']).T
                
                if self.mwe=='yes':
                    count_mwe = pd.DataFrame.from_dict(Counter(self.df[left_right].mwe.to_list()),
                                                       orient='index', columns=['mwe']).T
                    count_all = pd.concat([count_usas, count_pos, count_mwe])
                else:
                    count_all = pd.concat([count_usas, count_pos])
                
                count_all = count_all.fillna('-')
                stats_html = count_all.to_html(escape=False)
                stats_html = self.style+'<div class="dataframe-div">'+stats_html+"\n</div>"
                display(HTML(stats_html))
                
        # link the button with the function
        display_button.on_click(on_display_button_clicked)
        
        hbox1 = widgets.HBox([enter_text, text],
                             layout = widgets.Layout(height='35px'))
        hbox2 = widgets.HBox([filter_pos, select_pos],
                             layout = widgets.Layout(width='250px',
                                                     margin='0px 0px 0px 43px'))
        hbox3 = widgets.HBox([filter_usas, select_usas],
                             layout = widgets.Layout(width='300px',
                                                     margin='0px 0px 0px 13px'))
        if self.mwe=='yes':
            hbox4 = widgets.HBox([filter_mwe, select_mwe],
                                 layout = widgets.Layout(width='300px',
                                                         margin='0px 0px 0px 36px'))
            hbox5a = widgets.HBox([hbox2, hbox3])
            hbox5 = widgets.VBox([hbox5a, hbox4])
        else:
            hbox5 = widgets.HBox([hbox2, hbox3])
        hbox6 = widgets.HBox([display_button],
                             layout=Layout(margin= '0px 0px 10px 75px'))
        vbox = widgets.VBox([hbox1, hbox5, hbox6],
                             layout = widgets.Layout(width='500px'))
        
        return vbox, display_out, stat_out
    
    
    def display_two_tag_texts(self): 
        '''
        Function to display tagged texts commparison 
        '''
        # widget for displaying first text
        vbox1, display_out1, stat_out1 = self.display_tag_text('left')
        vbox2, display_out2, stat_out2 = self.display_tag_text('right')
        
        hbox = widgets.HBox([vbox1, vbox2])
        vbox = widgets.VBox([hbox, display_out1, display_out2, stat_out1, stat_out2])
        
        return vbox
        
        
    def top_entities(self, 
                     count_ent: dict, 
                     top_n: int) -> dict:
        '''
        Function to identify top entities in the text
        
        Args:
            count_ent: the count of the selected entity
            top_n: the number of top items to be shown
        '''
        # count and identify top entity
        top_ent = dict(sorted(count_ent.items(), key=lambda x: x[1], reverse=False)[-top_n:])
        
        return top_ent
        
        
    def count_entities(self, which_text: str, which_ent: str) -> dict:
        '''
        Function to count the number of selected entities in the text
        
        Args:
            which_ent: the selected entity to be counted
        '''
        if which_text=='all texts':
            df = self.tagged_df
        else:
            df = self.tagged_df[self.tagged_df['text_name']==which_text].reset_index(drop=True)
        
        # exclude punctuations
        items_to_exclude = ['PUNCT', ['PUNCT']]
        
        import itertools

        number_of_cpu = joblib.cpu_count()        
        # count entities based on type of entities
        if which_ent=='usas_tags' or which_ent=='usas_tags_def':
            # identify usas_tags or usas_tags_def
            print('Analysing text...')
            ent = [item for item in tqdm(list(itertools.chain(*df[which_ent].to_list()))) \
                   if item not in items_to_exclude]
            
        elif which_ent=='lemma' or which_ent=='pos' or which_ent=='token':
            # identify lemmas, tokens or pos tags
            ent = [item for n, item in enumerate(df[which_ent].to_list()) 
                   if df['pos'][n] not in items_to_exclude]
            
        elif which_ent=='mwe':
            # identify mwe indexes
            all_mwe = set(zip(df[df['mwe']=='yes']['start_index'],\
                              df[df['mwe']=='yes']['end_index']))
            
            # join the mwe expressions
            ent = [' '.join([self.tagged_df.loc[i,'token'] \
                             for i in range(mwe[0],mwe[1])]) for mwe in all_mwe]
        
        return Counter(ent)
    
    
    def count_text(self, which_text: str, which_ent, inc_ent):
        '''
        Function to identify texts based on selected top entities
        
        Args:
            which_ent: the selected entity, e.g., USAS tags, POS tags, etc.
            inc_ent: the included entity type, e.g., Z1, VERB, etc.
        '''
        if which_text=='all texts':
            df = self.tagged_df
        else:
            df = self.tagged_df[self.tagged_df['text_name']==which_text].reset_index(drop=True)
            
        # placeholder for selected texts
        selected_texts = []
        
        # iterate over selected entities and identified tokens based on entity type
        for n, tag in enumerate(df[which_ent]):
            if type(tag)==list:
                for i in tag:
                    if i in inc_ent:
                        selected_texts.append(df['token'][n])
            else:
                if tag in inc_ent:
                    selected_texts.append(df['token'][n])
                    
        return Counter(selected_texts)
    
    
    def visualize_stats(
            self, 
            which_text: str,
            top_ent: dict,
            sum_ent: int,
            top_n: int,
            title: str,
            color: str
            ):
        '''
        Create a horizontal bar plot for displaying top n named entities
        
        Args:
            top_ent: the top entities to display
            top_ent: the number of top entities to display
            title: title of the bar plot
            color: color of the bars
        '''
        if top_ent!={}:
            # specify the width, height and tick range for the plot
            display_height = top_n/2
            range_tick = max(1,round(max(top_ent.values())/5))
            
            # visualize the entities using horizontal bar plot
            fig = plt.figure(figsize=(10, max(5,display_height)))
            plt.barh(list(top_ent.keys()), 
                     list(top_ent.values()),
                     color=color)
            
            # display the values on the bars
            for i, v in enumerate(list(top_ent.values())):
                plt.text(v+(len(str(v))*0.05), 
                         i, 
                         '  {} ({:.0f}%)'.format(v,100*v/sum_ent), 
                         fontsize=10)
            
            # specify xticks, yticks and title
            plt.xticks(range(0, max(top_ent.values())+int(2*range_tick), 
                             range_tick), 
                       fontsize=10)
            plt.yticks(fontsize=10)
            bar_title = 'Top {} "{}" in text: "{}"'.format(min(top_n,
                                                           len(top_ent.keys())),
                                                             title, 
                                                             which_text)
            plt.title(bar_title, fontsize=12)
            plt.show()
            
        return fig, bar_title
        
        
    def analyse_tags(self):
        '''
        Function to display options for analysing entity/tag
        '''
        # options for bar chart titles
        titles = {'usas_tags': 'USAS tags',
                  'usas_tags_def': 'USAS tag definitions',
                  'pos': 'Part-of-Speech Tags',
                  'lemma': 'lemmas',
                  'token': 'tokens'}
        
        # entity options
        ent_options = ['usas_tags_def', #'usas_tags', 
                       'pos', 'lemma', 'token']
        
        if self.mwe=='yes':
            titles['mwe']='Multi-Word Expressions'
            ent_options.append('mwe')
        
        # placeholder for saving bar charts
        self.figs = []
        
        # widgets for selecting text_name to analyse
        choose_text, my_text = self.select_text_widget(entity=True)
        
        # widget to select entity options
        enter_entity, select_entity = self.select_options('<b>Select entity to show:</b>',
                                                        ent_options,
                                                        'usas_tags_def')
        
        # widget to select n
        enter_n, top_n = self.select_n_widget('<b>Select n:</b>', 5)
        
        # widget to analyse tags
        analyse_button, analyse_out = self.click_button_widget(desc='Show top entities',
                                                       margin='20px 0px 0px 0px',
                                                       width='155px')
        
        # function to define what happens when the button is clicked
        def on_analyse_button_clicked(_):
            # clear save_out
            with save_out:
                clear_output()
                
            # clear analyse_top_out
            with analyse_top_out:
                clear_output()
                
            # display bar chart for selected entity
            with analyse_out:
                clear_output()
                
                # get selected values
                which_text=my_text.value
                which_ent=select_entity.value
                n=top_n.value
                title=titles[which_ent]
                
                # get top entities
                count_ent = self.count_entities(which_text, which_ent)
                sum_ent = sum(count_ent.values())
                top_ent = self.top_entities(count_ent, n)
                
                # create bar chart
                fig, bar_title = self.visualize_stats(which_text,
                                                      top_ent,
                                                      sum_ent,
                                                      n,
                                                      title,
                                                      '#2eb82e')
                
                # append to self.figs for saving later
                self.figs.append([fig, bar_title])
                
                # update options for displaying tokens in entity type
                if which_ent!='mwe' and which_ent!='token':
                #if which_ent!='token':
                    new_options = list(top_ent.keys())
                    new_options.reverse()
                    select_text.options = new_options
                    select_text.value = [new_options[0]]
                    enter_text.value = '<b>Select {}:</b>'.format(titles[which_ent][:-1])
                else:
                    select_text.options = ['None']
                    select_text.value = ['None']
                    enter_text.value = '<b>No selection required.</b>'
        
        # link the button with the function
        analyse_button.on_click(on_analyse_button_clicked)
        
        # widget to select top entity type and display top tokens
        enter_text, select_text = self.select_multiple_options('<b>Select tag/lemma/token:</b>',
                                                               ['None'],
                                                               ['None'])
        
        # widget to select n
        enter_n_text, top_n_text = self.select_n_widget('<b>Select n:</b>', 5)
        
        # widget to analyse texts
        analyse_top_button, analyse_top_out = self.click_button_widget(desc='Show top words', 
                                                                       margin='12px 0px 0px 0px',
                                                                       width='155px')
        
        # function to define what happens when the button is clicked
        def on_analyse_top_button_clicked(_):
            # clear save_out
            with save_out:
                clear_output()
            
            # display bar chart for selected entity type
            with analyse_top_out:
                # obtain selected entity type
                which_ent=select_entity.value
                
                # only create new bar chart if not 'mwe' or 'token' (already displayed)
                if which_ent!='mwe' and which_ent!='token':
                #if which_ent!='token':
                    # get selected values
                    which_text=my_text.value
                    clear_output()
                    inc_ent=select_text.value
                    n=top_n_text.value
                    
                    # display bar chart for every selected entity type
                    for inc_ent_item in inc_ent:
                        title = inc_ent_item
                        count_ent = self.count_text(which_text, which_ent, inc_ent_item)
                        sum_ent = sum(count_ent.values())
                        top_text = self.top_entities(count_ent, n)
                        
                        try:
                            fig, bar_title = self.visualize_stats(which_text,
                                                                  top_text,
                                                                  sum_ent,
                                                                  n,
                                                                  title,
                                                                  '#008ae6')
                            self.figs.append([fig, bar_title])
                        except:
                            print('Please show top entities first!')
                else:
                    # display warning for 'mwe' or 'token'
                    with analyse_out:
                        print('The top {} are shown above!'.format(titles[which_ent]))
        
        # link the button with the function
        analyse_top_button.on_click(on_analyse_top_button_clicked)
        
        # widget to save the above
        save_button, save_out = self.click_button_widget(desc='Save analysis', 
                                                         margin='10px 0px 0px 0px',
                                                         width='155px')
        
        # function to define what happens when the save button is clicked
        def on_save_button_clicked(_):
            with save_out:
                clear_output()
                if self.figs!=[]:
                    # set the output folder for saving
                    out_dir='./output/'
                    
                    print('Analysis saved! Click below to download:')
                    # save the bar charts as jpg files
                    for fig, bar_title in self.figs:
                        bar_title = ' '.join(bar_title.split('/'))
                        bar_title = ' '.join(bar_title.split('"'))
                        file_name = '-'.join(bar_title.split()) + '.jpg'
                        fig.savefig(out_dir+file_name, bbox_inches='tight')
                        display(DownloadFileLink(out_dir+file_name, file_name))
                    
                    # reset placeholder for saving bar charts
                    self.figs = []
                else:
                    print('You need to generate the bar charts before you can save them!')
        
        # link the save_button with the function
        save_button.on_click(on_save_button_clicked)
        
        # displaying inputs, buttons and their outputs
        hbox1 = widgets.HBox([choose_text, my_text],
                             layout = widgets.Layout(height='35px'))
        vbox1 = widgets.VBox([enter_entity,
                              select_entity,
                              enter_n, top_n,], 
                             layout = widgets.Layout(width='250px', height='151px'))
        vbox2 = widgets.VBox([analyse_button,
                              save_button], 
                             layout = widgets.Layout(width='250px', height='100px'))
        vbox3 = widgets.VBox([vbox1, vbox2])
        vbox4 = widgets.VBox([enter_text, 
                              select_text, 
                              enter_n_text, top_n_text,
                              analyse_top_button],
                             layout = widgets.Layout(width='250px', height='250px'))
        
        hbox2 = widgets.HBox([vbox3, vbox4])
        
        vbox = widgets.VBox([hbox1, hbox2, save_out],
                            layout = widgets.Layout(width='500px'))
        
        return vbox, analyse_out, analyse_top_out
    
    
    def analyse_two_tags(self):
        '''
        Function to display options for comparing text analysis
        '''
        vbox1, analyse_out1, analyse_top_out1 = self.analyse_tags()
        vbox2, analyse_out2, analyse_top_out2 = self.analyse_tags()
        
        hbox = widgets.HBox([vbox1, vbox2])
        vbox1 = widgets.VBox([analyse_out1, analyse_top_out1])
        vbox2 = widgets.VBox([analyse_out2, analyse_top_out2])
        vbox = widgets.VBox([hbox, vbox1, vbox2])
        
        return vbox
    
    
    def save_to_csv(self, 
                    out_dir: str,
                    file_name: str):
        '''
        Function to save tagged texts to csv file
        
        Args:
            out_dir: the output file directory
            file_name: the name of teh saved file
        '''
        # split into chunks
        chunks = np.array_split(self.tagged_df.index, len(self.text_df)) 
        
        # save the tagged text into csv
        for chunck, subset in enumerate(tqdm(chunks)):
            if chunck == 0:
                self.tagged_df.loc[subset].to_csv(out_dir+file_name, 
                                                  mode='w', 
                                                  index=True)
            else:
                self.tagged_df.loc[subset].to_csv(out_dir+file_name, 
                                                  header=None, 
                                                  mode='a', 
                                                  index=True)
    
    
    def save_to_xml(self, 
                    out_dir: str,
                    file_name: str):
        '''
        Function to save tagged texts to zip of txt (pseudo-xml) file
        
        Args:
            out_dir: the output file directory
            file_name: the name of teh saved file
        '''
        # create a directory for saving .txt files
        os.makedirs('./output/saved_files', exist_ok=True)
        
        # save tagged texts
        for text in tqdm(self.tagged_df.text_name.unique()):
            this_text = []
            for n, row in enumerate(self.tagged_df[self.tagged_df['text_name']==text].itertuples()):
                if self.mwe=='yes':
                    pseudo_xml = '<w id="{}" pos="{}" sem="{}/{}" mwe="{}">{}</w>'.format(n, 
                                                                                          row.pos, 
                                                                                          row.usas_tags[0], 
                                                                                          row.usas_tags_def[0], 
                                                                                          row.mwe, 
                                                                                          row.token)
                else:
                    pseudo_xml = '<w id="{}" pos="{}" sem="{}/{}">{}</w>'.format(n, 
                                                                                 row.pos, 
                                                                                 row.usas_tags[0], 
                                                                                 row.usas_tags_def[0], 
                                                                                 row.token)
                this_text.append(pseudo_xml)
            with open('./output/saved_files/{}.txt'.format(text), 'w') as f:
                f.write('\n'.join(this_text))
        
        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file), 
                               os.path.relpath(os.path.join(root, file), 
                                               os.path.join(path, '..')))
        
        with zipfile.ZipFile(out_dir+file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipdir('./output/saved_files/', zipf)

        # remove files and directory once finished
        os.system('rm -r ./output/saved_files')
        
        
    def save_to_excel(self, 
                      out_dir: str,
                      file_name: str):
        '''
        Function to save analysis into an excel spreadsheet and download to local computer

        Args:
            out_dir: the name of the output directory.
            file_name: the name of the saved file 
        '''
        # save tagged texts onto an Excel spreadsheet
        values = [self.tagged_df.columns] + list(self.tagged_df.values)
        wb = Workbook()
        wb.new_sheet(sheet_name='tagged_texts', data=values)
        wb.save(out_dir + file_name)
        
    
    def save_options(self):
        '''
        options for saving tagged texts
        '''
        warnings.filterwarnings("ignore")
        
        # widget to select save options
        enter_save, select_save = self.select_options('<b>Select saving file type:</b>',
                                                      ['excel', 'csv', 'pseudo-XML'],
                                                      'excel')
        
        # widget to process texts
        process_button, process_out = self.click_button_widget(desc='Save tagged texts', 
                                                       margin='10px 0px 0px 0px',
                                                       width='160px')
        
        # function to define what happens when the top button is clicked
        def on_process_button_clicked(_):
            with process_out:
                clear_output()
                save_type = select_save.value
                out_dir = './output/'
                
                print('Saving tagged texts in progress. Please be patient...')

                if save_type =='excel':
                    file_name = 'tagged_texts.xlsx'
                    self.save_to_excel(out_dir, file_name)
                elif save_type =='csv':
                    file_name = 'tagged_texts.csv'
                    self.save_to_csv(out_dir, file_name)
                else:
                    file_name = 'tagged_texts.zip'
                    self.save_to_xml(out_dir, file_name)
                    
                # download the saved file onto your computer
                print('Tagged texts saved. Click below to download:')
                display(DownloadFileLink(out_dir+file_name, file_name))
                    
        # link the top_button with the function
        process_button.on_click(on_process_button_clicked)
        
        # displaying inputs, buttons and their outputs
        vbox1 = widgets.VBox([enter_save, select_save], 
                             layout = widgets.Layout(width='600px', height='80px'))
        vbox2 = widgets.VBox([process_button, process_out],
                             layout = widgets.Layout(width='600px'))
        
        vbox = widgets.VBox([vbox1, vbox2])
        
        return vbox
    
    
    def select_text_widget(self, entity: bool=False):
        '''
        Create widgets for selecting text to display
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value='<b>Select text:</b>',
            placeholder='',
            description=''
            )
        
        # use text_name for text_options
        text_options = self.text_df.text_name.to_list() # get the list of text_names
        
        # the option to select 'all texts' for analysing top entities
        if entity:
            text_options.insert(0, 'all texts')
        
        # widget to display text_options
        text = widgets.Combobox(
            placeholder='Choose text to display...',
            options=text_options,
            description='',
            ensure_option=True,
            disabled=False,
            layout = widgets.Layout(width='195px')
        )
        
        return enter_text, text
    
    
    def select_options(self, 
                       instruction: str,
                       options: list,
                       value: str):
        '''
        Create widgets for selecting the number of entities to display
        
        Args:
            instruction: text instruction for user
            options: list of options for user
            value: initial value of the widget
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widget to select entity options
        select_option = widgets.Dropdown(
            options=options,
            value=value,
            description='',
            disabled=False,
            layout = widgets.Layout(width='150px')
            )
        
        return enter_text, select_option
    
    
    def select_multiple_options(self, 
                                instruction: str,
                                options: list,
                                value: list):
        '''
        Create widgets for selecting muyltiple options
        
        Args:
            instruction: text instruction for user
            options: list of options for user
            value: initial value of the widget
        '''
        # widget to display instruction
        enter_m_text = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widget to select entity options
        select_m_option = widgets.SelectMultiple(
            options=options,
            value=value,
            description='',
            disabled=False,
            layout = widgets.Layout(width='150px')
            )
        
        return enter_m_text, select_m_option
        
        
    def select_n_widget(self, 
                        instruction: str, 
                        value: int,
                        max_v: int=1e+3):
        '''
        Create widgets for selecting the number of entities to display
        
        Args:
            instruction: text instruction for user
            value: initial value of the widget
        '''
        # widget to display instruction
        enter_n = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widgets for selecting n
        n_option = widgets.BoundedIntText(
            value=value,
            min=0,
            max=max_v,
            step=5,
            description='',
            disabled=False,
            layout = widgets.Layout(width='150px')
        )
        
        return enter_n, n_option
    
    
    def click_button_widget(
            self, 
            desc: str, 
            margin: str='10px 0px 0px 10px',
            width='320px'
            ):
        '''
        Create a widget to show the button to click
        
        Args:
            desc: description to display on the button widget
            margin: top, right, bottom and left margins for the button widget
            width: the width of the button
        '''
        # widget to show the button to click
        button = widgets.Button(description=desc, 
                                layout=Layout(margin=margin, width=width),
                                style=dict(font_weight='bold'))
        
        # the output after clicking the button
        out = widgets.Output()
        
        return button, out