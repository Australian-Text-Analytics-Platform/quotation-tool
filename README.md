# Quotation Tool

<b>Abstract:</b> This QuotationTool can be used to extract quotes from a text. In addition to extracting the quotes, the tool also provides information about who the speakers are, the location of the quotes (and the speakers) within the text, the identified named entities, etc., which can be useful for your text analysis.

## Setup
This tool has been designed for use with minimal setup from users. You are able to run it in the cloud and any dependencies with other packages will be installed for you automatically. In order to launch and use the tool, you just need to click the below icon.

[![Binder](https://binderhub.atap-binder.cloud.edu.au/badge_logo.svg)](https://binderhub.atap-binder.cloud.edu.au/v2/gh/Australian-Text-Analytics-Platform/quotation-tool/main?labpath=quote_extractor_notebook.ipynb)  

<b>Note:</b> CILogon authentication is required. You can use your institutional, Google or Microsoft account to login. If you have trouble authenticating, please refer to the [CILogon troubleshooting guide](documents/cilogon-troubleshooting.pdf).

If you do not have access to any of the above accounts, you can use the below link to access the tool (this is a free Binder version, limited to 2GB memory only).   

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Australian-Text-Analytics-Platform/quotation-tool/main?labpath=quote_extractor_notebook.ipynb)

It may take a few minutes for Binder to launch the notebook and install the dependencies for the tool. Please be patient.

## User Guide

For instructions on how to use the Quotation Tool, please refer to the [Quotation Tool User Guide](documents/quotation_help_pages.pdf).

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

## Citation
If you find the Quotation Tool useful in your research, please cite the following:  

Jufri, Sony & Sun, Chao (2022). Quotation Tool. v1.0. Australian Text Analytics Platform. Software. https://github.com/Australian-Text-Analytics-Platform/quotation-tool
