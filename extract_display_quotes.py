#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jul  4 14:55:46 2022

@author: sjufri

This code has been adapted (with permission) from the GenderGapTracker GitHub page 
(https://github.com/sfu-discourse-lab/GenderGapTracker/tree/master/NLP/main)
and modified to run on a Jupyter Notebook.

The quotation tool’s accuracy rate is evaluated in the below article:
The Gender Gap Tracker: Using Natural Language Processing to measure gender bias in media
(https://doi.org/10.1371/journal.pone.0245533)
'''

# import required packages
import os
import io
import sys
import codecs
import logging
import traceback
from collections import Counter
from datetime import datetime
import hashlib
from tqdm import tqdm

# matplotlib: visualization tool
from matplotlib import pyplot as plt

# pandas: tools for data processing
import pandas as pd

# spaCy and NLTK: natural language processing tools for working with language/text data
import spacy
from spacy import displacy
from spacy.tokens import Span
import nltk
nltk.download('punkt')
from nltk import Tree
from nltk.tokenize import sent_tokenize

# ipywidgets: tools for interactive browser controls in Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display, Markdown, clear_output, FileLink

# clone the GenderGapTracker GitHub page
path  = './'
clone = 'git clone https://github.com/sfu-discourse-lab/GenderGapTracker'
os.chdir(path)
os.system(clone)

# import the quote extractor tool
from config import config
sys.path.insert(0,'./GenderGapTracker/NLP/main')
from quote_extractor import extract_quotes
import utils


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
        

class QuotationTool():
    '''
    Interactive tool for extracting and displaying quotes in a text
    '''
    
    def __init__(self):
        '''
        Initiate the QuotationTool
        '''
        # initiate the app_logger
        self.app_logger = utils.create_logger('quote_extractor', log_dir='logs', 
                                         logger_level=logging.INFO,
                                         file_log_level=logging.INFO)
        
        # download spaCy's en_core_web_lg, the pre-trained English language tool from spaCy
        print('Loading spaCy language model...')
        print('This may take a while...')
        self.nlp = spacy.load('en_core_web_lg')
        print('Finished loading.')
        
        # initiate variables to hold texts and quotes in pandas dataframes
        self.text_df = None
        self.quotes_df = None
        
        # initiate the variables for file uploading
        self.file_uploader = widgets.FileUpload(
            description='Upload your files (txt, csv or xlsx)',
            accept='.txt, .xlsx, .csv ', # accepted file extension 
            multiple=True,  # True to accept multiple files
            error='File upload unsuccessful. Please try again!',
            layout = widgets.Layout(width='320px')
            )
    
        self.upload_out = widgets.Output()
        
        # give notification when file is uploaded
        def _cb(change):
            with self.upload_out:
                # clear output and give notification that file is being uploaded
                clear_output()
                
                # reading uploaded files
                self.process_upload()
                
                # give notification when uploading is finished
                print('Finished uploading files.')
                print('Currently {} text documents are loaded for analysis'.format(self.text_df.shape[0]))
            
        # observe when file is uploaded and display output
        self.file_uploader.observe(_cb, names='data')
        self.upload_box = widgets.VBox([self.file_uploader, self.upload_out])
        
        # initiate other required variables
        self.html = None
        self.figs = None
        self.current_text = None
        
        # create an output folder if not already exist
        os.makedirs('output', exist_ok=True)


    def load_txt(self, value: dict) -> list:
        '''
        Load individual txt file content and return a dictionary object, 
        wrapped in a list so it can be merged with list of pervious file contents.
        
        Args:
            value: the file containing the text data
        '''
        temp = {'text_name': value['metadata']['name'][:-4],
                'text': codecs.decode(value['content'], encoding='utf-8', errors='replace')
        }
    
        return [temp]


    def load_table(self, value: dict, file_fmt: str) -> list:
        '''
        Load csv or xlsx file
        
        Args:
            value: the file containing the text data
            file_fmt: the file format, i.e., 'csv', 'xlsx'
        '''
        # read the file based on the file format
        if file_fmt == 'csv':
            temp_df = pd.read_csv(io.BytesIO(value['content']))
        if file_fmt == 'xlsx':
            temp_df = pd.read_excel(io.BytesIO(value['content']))
            
        # check if the column text and text_name present in the table, if not, skip the current spreadsheet
        if ('text' not in temp_df.columns) or ('text_name' not in temp_df.columns):
            print('File {} does not contain the required header "text" and "text_name"'.format(value['metadata']['name']))
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


    def nlp_preprocess(self, text):
        '''
        Pre-process text and fit it with Spacy language model into the column "spacy_text"

        Args:
            temp_df: the temporary pandas dataframe containing the text data
        '''
        text = sent_tokenize(text)
        text = ' '.join(text)
        text = utils.preprocess_text(text)
        try: 
            text = self.nlp(text)
        except:
            print('The below text is too large. Consider breaking it down into smaller texts .')
            print(text[:20])
            
        return text
    

    def process_upload(self, deduplication: bool = True):    
        '''
        Pre-process uploaded .txt files into pandas dataframe

        Args:
            deduplication: option to deduplicate text_df by text_id
        '''
        # create an empty list for a placeholder to store all the texts
        all_data = []
        
        # read and store the uploaded files
        files = list(self.file_uploader.value.keys())
        
        print('Reading uploaded files...')
        print('This may take a while...')
        for file in tqdm(files):
            if file.lower().endswith('txt'):
                text_dic = self.load_txt(self.file_uploader.value[file])
            else:
                text_dic = self.load_table(self.file_uploader.value[file], \
                    file_fmt=file.lower().split('.')[-1])
            all_data.extend(text_dic)
        
        # convert them into a pandas dataframe format, add unique id and pre-process text
        self.text_df = pd.DataFrame.from_dict(all_data)
        self.text_df = self.hash_gen(self.text_df)
        
        # deduplicate the text_df by text_id
        if deduplication:
            self.text_df.drop_duplicates(subset='text_id', keep='first', inplace=True)
        print('step 4 completed')
    
    
    def extract_inc_ent(
            self, 
            list_of_string: list, 
            spacy_doc: spacy.tokens.doc.Doc, 
            inc_ent: list
            ) -> list:
        '''
        Extract included named entities from a list of string

        Args:
            list_of_string: a list of string from which to extract the named entities
            spacy_doc: spaCy's processed text for the above list of string
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        
        return [
            [(str(ent), ent.label_) for ent in spacy_doc.ents \
                if (str(ent) in string) & (ent.label_ in inc_ent)]\
                    for string in list_of_string
                    ]
        

    def get_quotes(self, inc_ent: list) -> pd.DataFrame:
        '''
        Extract quotes and their meta-data (quote_id, quote_index, etc.) from the text
        and return as a pandas dataframe

        Args:
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        print('Extracting quotes...')
        print('This may take a while...')
        # create an empty list to store all detected quotes
        all_quotes = []
        
        # go through all the texts and start extracting quotes
        for row in tqdm(self.text_df.itertuples(), total=len(self.text_df)):
            text_id = row.text_id
            text_name = row.text_name
            doc = self.nlp_preprocess(row.text)
            
            try:        
                # extract the quotes
                quotes = extract_quotes(doc_id=text_id, doc=doc, 
                                        write_tree=False)
                
                # extract the named entities
                speaks, qts = [quote['speaker'] for quote in quotes], [quote['quote'] for quote in quotes]
                speak_ents = self.extract_inc_ent(speaks, doc, inc_ent)
                quote_ents = self.extract_inc_ent(qts, doc, inc_ent)
        
                # add text_id, quote_id and named entities to each quote
                for n, quote in enumerate(quotes):
                    quote['text_id'] = text_id
                    quote['text_name'] = text_name
                    quote['quote_id'] = str(n)
                    quote['speaker_entities'] = list(set(speak_ents[n]))
                    quote['quote_entities'] = list(set(quote_ents[n]))
                    
                # store them in all_quotes
                all_quotes.extend(quotes)
                    
            except:
                # this will provide some information in the case of an error
                self.app_logger.exception("message")
                traceback.print_exception()
            
                
        # convert the outcome into a pandas dataframe
        self.quotes_df = pd.DataFrame.from_dict(all_quotes)
        
        # convert the string format quote spans in the index columns to a tuple of integers
        for column in self.quotes_df.columns:
            if column.endswith('_index'):
                self.quotes_df[column].replace('','(0,0)', inplace=True)
                self.quotes_df[column] = self.quotes_df[column].apply(eval)
        
        # re-arrange the columns
        new_index = ['text_id', 'text_name', 'quote_id', 'quote', 'quote_index', 'quote_entities', 
                     'speaker', 'speaker_index', 'speaker_entities',
                     'verb', 'verb_index', 'quote_token_count', 'quote_type', 'is_floating_quote']
        self.quotes_df = self.quotes_df.reindex(columns=new_index)
                
        return self.quotes_df
    
    
    def add_entities(
            self, 
            spacy_doc: spacy.tokens.doc.Doc, 
            selTokens: list, 
            inc_ent: list
            ) -> list:
        '''
        Add included named entities to displaCy code

        Args:
            spacy_doc: spaCy's processed text for the above list of string
            selTokens: options to display speakers, quotes or named entities
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        # empty list to hold entities code
        ent_code_list = []
        
        # create span code for entities
        for ent in spacy_doc.ents:
            if (ent.start in selTokens) & (ent.label_ in inc_ent):
                span_code = "Span(doc, {}, {}, '{}'),".format(ent.start, 
                                                  ent.end, 
                                                  ent.label_) 
                ent_code_list.append(span_code)
        
        # combine codes for all entities
        ent_code = ''.join(ent_code_list)
        
        return ent_code
    
    
    def show_quotes(
            self, 
            text_name: str, 
            show_what: list, 
            inc_ent: list
            ):
        '''
        Display speakers, quotes and named entities inside the text using displaCy

        Args:
            text_name: the text_name of the text you wish to display
            show_what: options to display speakers, quotes or named entities
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        # formatting options
        TPL_SPAN = '''
        <span style="font-weight: bold; display: inline-block; position: relative; 
        line-height: 55px">
            {text}
            {span_slices}
            {span_starts}
        </span>
        '''
        
        TPL_SPAN_SLICE = '''
        <span style="background: {bg}; top: {top_offset}px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
        </span>
        '''
        
        TPL_SPAN_START = '''
        <span style="background: {bg}; top: {top_offset}px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
            <span style="background: {bg}; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
                {label}{kb_link}
            </span>
        </span>
        '''
        
        colors = {'QUOTE': '#66ccff', 'SPEAKER': '#66ff99'}
        options = {'ents': ['QUOTE', 'SPEAKER'], 
                   'colors': colors, 
                   'top_offset': 42,
                   'template': {'span':TPL_SPAN,
                               'slice':TPL_SPAN_SLICE,
                               'start':TPL_SPAN_START},
                   'span_label_offset': 14,
                   'top_offset_step':14}
        
        # get the spaCy text 
        current_text = self.text_df[self.text_df['text_name']==text_name]['text'].to_list()[0]
        doc = self.nlp_preprocess(current_text)
        
        # create a mapping dataframe between the character index and token index from the spacy text.
        loc2tok_df = pd.DataFrame([(t.idx, t.i) for t in doc], columns = ['loc', 'token'])
    
        # get the quotes and speakers indexes
        locs = {
            'QUOTE': self.quotes_df[self.quotes_df['text_name']==text_name]['quote_index'].tolist(),
            'SPEAKER': set(self.quotes_df[self.quotes_df['text_name']==text_name]['speaker_index'].tolist())
        }
    
        # create the displaCy code to visualise quotes and speakers
        my_code_list = ['doc.spans["sc"] = [', ']']
        
        for key in locs.keys():
            for loc in locs[key]:
                if loc!=(0,0):
                    # Find out all token indices that falls within the given span (variable loc)
                    selTokens = loc2tok_df.loc[(loc[0]<=loc2tok_df['loc']) & (loc2tok_df['loc']<loc[1]), 'token'].tolist()
                    
                    # option to display named entities only
                    if show_what==['NAMED ENTITIES']:
                        ent_code = self.add_entities(doc, selTokens, inc_ent)
                        my_code_list.insert(1,ent_code)
                    
                    # option to display speaker and/or quotes and/or named entities
                    elif key in show_what:
                        if 'NAMED ENTITIES' in show_what:
                            ent_code = self.add_entities(doc, selTokens, inc_ent)
                            my_code_list.insert(1,ent_code)
                        
                        start_token, end_token = selTokens[0], selTokens[-1] 
                        span_code = "Span(doc, {}, {}, '{}'),".format(start_token, end_token+1, key) 
                        my_code_list.insert(1,span_code)
                    
        # combine all codes
        my_code = ''.join(my_code_list)
    
        # execute the code
        exec(my_code)
        
        # display the preview in this notebook
        displacy.render(doc, style='span', options=options, jupyter=True)
        self.html = displacy.render(doc, style='span', options=options, jupyter=False, page=True)
        
    
    def analyse_quotes(self, inc_ent: list):
        '''
        Interactive tool to display and analyse speakers, quotes and named entities inside the text

        Args:
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        # widgets to select text_name to preview
        enter_text, text = self.select_text_widget(entity=False)
        
        # widgets to select which entities to preview, i.e., speakers and/or quotes and/or named entities
        entity_options, speaker_box, quote_box, ne_box = self.select_entity_widget(entity=True)
        
        # widgets to show the preview
        preview_button, preview_out = self.click_button_widget(desc='Preview', 
                                                               margin='10px 0px 0px 10px',
                                                               width='200px')
        
        # function to define what happens when the preview button is clicked
        def on_preview_button_clicked(_):
            # what happens when we click the preview_button
            with save_out:
                clear_output()
            
            with preview_out:                                                                                   
                clear_output()
                text_name = text.value
                
                # add the selected entities to display
                show_what = []
                if speaker_box.value:
                    show_what.append('SPEAKER')
                if quote_box.value:
                    show_what.append('QUOTE')
                if ne_box.value:
                    show_what.append('NAMED ENTITIES')
                
                # provide information in the case no entity is selected
                if show_what==[]:
                    print('Please select the entities to display!')
                else:
                    try:
                        # display the text and the selected entities
                        self.show_quotes(text_name, show_what, inc_ent)
                    except:
                        # provide information in the case no text is selected
                        print('Please select the text to preview')
        
        # link the preview_button with the function
        preview_button.on_click(on_preview_button_clicked)
        
        # widget to save the above preview
        save_button, save_out = self.click_button_widget(desc='Save Preview', 
                                                         margin='10px 0px 0px 10px',
                                                         width='200px')
        
        # function to define what happens when the save button is clicked
        def on_save_button_clicked(_):
            with save_out:
                try:
                    # set the output folder for saving
                    out_dir='./output/'
                    text_name = text.value
                    file_name = '-'.join(text_name.split()) +'.html'
                    
                    # save the preview as an html file
                    file = open(out_dir+str(text_name)+'.html', 'w')
                    file.write(self.html)
                    file.close()
                    clear_output()
                    print('Preview saved! Click below to download:')
                    display(DownloadFileLink(out_dir+str(text_name)+'.html', file_name))
                except:
                    print('You need to generate a preview before you can save it!')
        
        # link the save_button with the function
        save_button.on_click(on_save_button_clicked)
        
        # widgets for displaying inputs, buttons and outputs
        vbox2 = widgets.VBox([enter_text, text], 
                             layout = widgets.Layout(width='300px'))
        vbox1 = widgets.VBox([entity_options, speaker_box, quote_box, ne_box],
                             layout = widgets.Layout(width='300px'))
        
        hbox = widgets.HBox([vbox1, vbox2])
        vbox = widgets.VBox([hbox, preview_button, save_button, save_out, preview_out])
        
        return vbox
    
    
    def analyse_entities(self, inc_ent: list):
        '''
        Interactive tool to display and analyse named entities inside the text

        Args:
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        # widgets for selecting text_name to analyse
        enter_text, text = self.select_text_widget(entity=True)
        
        # widgets for selecting whether to display entities in speakers and/or quotes
        entity_options, speaker_box, quote_box, ne_box = self.select_entity_widget()
        
        # widgets for selecting whether to display entity names and/or types
        label_options, name_box, entity_box = self.name_or_type_widget()
        
        # widgets for selecting the number of entities to display
        enter_n, top_n_option = self.select_n_widget()

        # widget to show top entities
        top_button, top_out = self.click_button_widget(desc='Show Top Entities', 
                                                       margin='10px 0px 0px 30px',
                                                       width='200px')
        
        # function to define what happens when the top button is clicked
        def on_top_button_clicked(_):
            with save_out:
                clear_output()
            with top_out:
                clear_output()
                text_name = text.value
                top_n = top_n_option.value
                
                # process the selected quote and/or speaker entities
                which_ents=[]; ent_types=[]
                if quote_box.value:
                    which_ents.append('quote_entities')
                if speaker_box.value:
                    which_ents.append('speaker_entities')
                if which_ents==[]:
                    print('Please select whether to display entities in the speakers and/or quotes!')
                
                # process the selected entity names and/or types
                if name_box.value:
                    ent_types.append('name')
                if entity_box.value:
                    ent_types.append('label')
                if ent_types==[]:
                    print('Please select whether to display entity names and/or types!')
                
                self.figs = []
                # display the selections
                for ent_type in ent_types:
                    for which_ent in which_ents:
                        try:
                            fig, bar_title = self.top_entities(text_name, which_ent, ent_type, top_n)
                            self.figs.append([fig, bar_title])
                        except:
                            if text_name=='':
                                print('Please select the text to analyse')

        # link the top_button with the function
        top_button.on_click(on_top_button_clicked)
        
        # widget to save the above preview
        save_button, save_out = self.click_button_widget(desc='Save Top Entities', 
                                                         margin='10px 0px 0px 30px',
                                                         width='200px')
        
        # function to define what happens when the save button is clicked
        def on_save_button_clicked(_):
            with save_out:
                if self.figs!=[]:
                    # set the output folder for saving
                    out_dir='./output/'
                    
                    print('Top entities saved! Click below to download:')
                    # save the top entities as jpg files
                    for fig, bar_title in self.figs:
                        file_name = '-'.join(bar_title.split()) + '.jpg'
                        fig.savefig(out_dir+file_name, bbox_inches='tight')
                        display(DownloadFileLink(out_dir+file_name, file_name))
                else:
                    print('You need to generate the bar charts before you can save them!')
        
        # link the save_button with the function
        save_button.on_click(on_save_button_clicked)
        
        # displaying inputs, buttons and their outputs
        vbox1 = widgets.VBox([enter_text, text], 
                             layout = widgets.Layout(width='290px'))
        vbox2 = widgets.VBox([entity_options, speaker_box, quote_box], 
                             layout = widgets.Layout(width='300px', height='100px'))
        vbox3 = widgets.VBox([label_options, name_box, entity_box], 
                             layout = widgets.Layout(width='350px'))
        vbox4 = widgets.VBox([enter_n, top_n_option], 
                             layout = widgets.Layout(width='250px', height='80px'))
        vbox5 = widgets.VBox([top_button, save_button], 
                             layout = widgets.Layout(width='250px', height='80px'))
        
        hbox1 = widgets.HBox([vbox2, vbox3, vbox4])
        hbox2 = widgets.HBox([vbox1, vbox5],
                             layout=Layout(margin='5px 0px 20px 0px'))
        
        vbox = widgets.VBox([hbox1, hbox2, save_out, top_out])
        
        return vbox
    
    
    def top_entities(
            self, 
            text_name: str, 
            which_ent: str, 
            ent_type: list, 
            top_n: int=5
            ):
        '''
        Display top n named entities identified in the speakrs and/or quotes

        Args:
            text_name: the text_name of the text you wish to display
            which_ent: option to display named entities in speakers and/or quotes
            ent_type: choose whether to display entity names or types
            top_n: the number of top entities to display
        '''
        # specify the text to analyse ('all texts' or each text individually)
        if text_name=='all texts':
            most_ent = self.quotes_df[which_ent].to_list()
        else:
            most_ent = self.quotes_df[self.quotes_df['text_name']==text_name][which_ent].tolist()
        
        # get the top n entities from the selected text
        most_ent = list(filter(None,most_ent))
        most_ent = [ent for most in most_ent for ent in most]
        if ent_type=='name':
            most_ent = Counter([ent_name for ent_name, ent_label in most_ent])
        if ent_type=='label':
            most_ent = Counter([ent_label for ent_name, ent_label in most_ent])        
        top_ent = dict(sorted(most_ent.items(), key=lambda x: x[1], reverse=False)[-top_n:])
        
        # visualize them
        fig, bar_title = self.visualize_entities(text_name, which_ent, ent_type, top_n, top_ent)
        
        return fig, bar_title
    
    
    def visualize_entities(
            self, 
            text_name: str, 
            which_ent: str, 
            ent_type: str, 
            top_n: int, 
            top_ent: dict
            ):
        '''
        Create a horizontal bar plot for displaying top n named entities in the speakrs and/or quotes

        Args:
            text_name: the text_name of the text you wish to display
            which_ent: option to display named entities in speakers and/or quotes
            ent_type: choose whether to display entity names or types
            top_n: the number of top entities to display
            top_ent: the top entities to display
        '''
        if top_ent!={}:
            # define color formatting and option for entity names/types            
            bar_colors = {'speaker_entities':'#2eb82e',
                          'quote_entities':'#008ae6'}
            ent_types = {'name': 'entity names',
                         'label': 'entity types'}
            
            # specify the width, height and tick range for the plot
            display_width = top_n/1.5
            display_height = top_n/2
            range_tick = max(1,round(max(top_ent.values())/5))
            
            # visualize the entities using horizontal bar plot
            fig = plt.figure(figsize=(10, max(5,display_height)))
            plt.barh(list(top_ent.keys()), list(top_ent.values()), color=bar_colors[which_ent])
            
            # display the values on the bars
            for i, v in enumerate(list(top_ent.values())):
                plt.text(v+(len(str(v))*0.05), i, str(v), fontsize=12)
            
            # specify xticks, yticks and title
            plt.xticks(range(0, max(top_ent.values())+range_tick, range_tick), fontsize=12)
            plt.yticks(fontsize=12)
            bar_title = 'Top {} {} entities ({}) in {}'.format(min(top_n,len(top_ent.keys())),
                                                             which_ent[:-9],
                                                             ent_types[ent_type],
                                                             text_name)
            plt.title(bar_title, fontsize=14)
            plt.show()
            
            return fig, bar_title
            
        elif text_name!='':
            print('No entities identified in the {}s.'.format(which_ent[:-9]))
        
        
    def select_text_widget(self, entity: bool=False):
        '''
        Create widgets for selecting text_name to analyse

        Args:
            entity: option to include 'all texts' for analysing top entities
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value='<b>Select the text to analyse:</b>',
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
            placeholder='Choose text to analyse...',
            options=text_options,
            description='',
            ensure_option=True,
            disabled=False,
            layout = widgets.Layout(width='195px')
        )
        
        return enter_text, text
    
    
    def select_entity_widget(self, entity: bool=False):
        '''
        Create widgets for selecting which entities to preview, 
        i.e., speakers and/or quotes and/or named entities

        Args:
            entity: option to include a check box for displaying named entities
        '''
        ne_box = None
        
        # widget to display instruction
        entity_options = widgets.HTML(
            value="<b>Select which entity to show:</b>",
            placeholder='',
            description='',
            )
        
        # widget to display speaker check box
        speaker_box = widgets.Checkbox(
            value=False,
            description='Speaker',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        # widget to display speaker quote box
        quote_box = widgets.Checkbox(
            value=False,
            description='Quote',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        # widget to display named entity check box
        if entity:
            ne_box = widgets.Checkbox(
                value=False,
                description='Named Entities',
                disabled=False,
                indent=False,
                layout=Layout(margin='0px 0px 0px 0px')
                )
        
        return entity_options, speaker_box, quote_box, ne_box
    
        
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
        
    
    def name_or_type_widget(self):
        '''
        Create widgets for selecting whether to display entity names and/or types
        '''
        # widget to display instruction
        label_options = widgets.HTML(
            value="<b>Display entity names and/or types:</b>",
            placeholder='',
            description='',
            )
        
        # widget to display entity names box
        name_box = widgets.Checkbox(
            value=False,
            description='Entity names (e.g., John Doe, Sydney, etc.)',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        # widget to display entity types box
        entity_box = widgets.Checkbox(
            value=False,
            description='Entity types (e.g., PERSON, LOC, etc.)',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        return label_options, name_box, entity_box
    
    
    def select_n_widget(self):
        '''
        Create widgets for selecting the number of entities to display
        '''
        # widget to display instruction
        enter_n = widgets.HTML(
            value='<b>The number of top entities to display:</b>',
            placeholder='',
            description=''
            )
        
        # widgets for selecting the number of top entities
        top_n_option = widgets.BoundedIntText(
            value=5,
            min=0,
            step=5,
            description='',
            disabled=False,
            layout = widgets.Layout(width='150px')
        )
        
        return enter_n, top_n_option