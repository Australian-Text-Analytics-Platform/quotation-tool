# Quotation Tool

<b>Abstract:</b> This QuotationTool can be used to extract quotes from a text. In addition to extracting the quotes, the tool also provides information about who the speakers are, the location of the quotes (and the speakers) within the text, the identified named entities, etc., which can be useful for your text analysis.

## Setup
This tool has been designed for use with minimal setup from users. You are able to run it in the cloud and any dependencies with other packages will be installed for you automatically. In order to launch and use the tool, you just need to click the below icon.

1. This is the preferred link, CILogon authentication is required where you can sign in with your institutional logon or Google/Microsoft account.
[![Binder](https://binderhub.atap-binder.cloud.edu.au/badge_logo.svg)](https://binderhub.atap-binder.cloud.edu.au/v2/gh/Australian-Text-Analytics-Platform/quotation-tool/GGT_update_20220915?labpath=quote_extractor_notebook.ipynb)  

If you are unable to access the tool via the first link above, then use the second link below. This is the free version of Binder, with less CPU and memory capacity (up to 2GB only).  

2. This link is for people without Australian institutional affiliations  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Australian-Text-Analytics-Platform/quotation-tool/GGT_update_20220915?labpath=quote_extractor_notebook.ipynb)  

<b>Note:</b> this may take a few minutes to launch as Binder needs to install the dependencies for the tool.

### Setting up on your own computer

If you know your way around the command line and are comfortable installing software, you might want to set up your own computer to run this notebook. 

Firstly, you need to install the [Anaconda Python distribution](https://www.anaconda.com/products/distribution) (You may also need to install [Git](https://github.com/git-guides/install-git#:~:text=Git%20can%20be%20installed%20on,most%20Mac%20and%20Linux%20machines!) if you are on Windows). Then, open your terminal (on MacOS) or your Git command line (on Windows) and follow the below steps to set up an environment with all the required packages:

* Clone the repository: git clone https://github.com/Australian-Text-Analytics-Platform/quotation-tool
* Change to the 'quote_tool' directory: cd quote_tool
* Create the environment: conda env create -f environment.yml
* Activate the environment: conda activate quote_tool
* Run Jupyter notebook: jupyter notebook quote_extractor_notebook.ipynb

## Load the data
<table style='margin-left: 10px'><tr>
<td> <img width='45' src='./img/txt_icon.png'/> </td>
<td> <img width='45' src='./img/xlsx_icon.png'/> </td>
<td> <img width='45' src='./img/csv_icon.png'/> </td>
<td> <img width='45'src='./img/zip_icon.png'/> </td>
</tr></table>

Using this tool, you can extract quotes directly from a text file (or a number of text files). Alternatively, you can also extract quotes from a text column inside your excel spreadsheet. You just need to upload your files (.txt, .xlsx or .csv) and access them via the Notebook.  

<b>Note:</b> If you have a large number of text files (more than 10MB in total), we suggest you compress (zip) them and upload the zip file instead. If you need assistance on how to compress your file, please check [the user guide](https://github.com/Australian-Text-Analytics-Platform/quotation-tool/blob/main/documents/jupyter-notebook-guide.pdf).  


## Extract and Display the Quotes
Once your files have been uploaded, you can use the QuotationTool to extract quotes from the text. The quotes, along with their metadata, will be stored in a table format inside a pandas dataframe. 

<img width='740' src='./img/quotes_df.png'/> 

Additionally, using the interactive tool, you can display the text, along with the extracted quotes, speakers and named entities, on the Notebook for further analysis.

<img width='740' src='./img/quote_display.png'/>

## Reference
This code has been adapted (with permission) from the [GenderGapTracker GitHub page](https://github.com/sfu-discourse-lab/GenderGapTracker/tree/master/NLP/main) and modified to run on a Jupyter Notebook. The quotation tool’s accuracy rate is evaluated in [this article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0245533).
