# DIDit

DIDit is a Python module specifically designed to analyze textual time series data and detect changes in identity using a signals intelligence approach. The module's development was driven by the author's desire to create a unique tool that provides insights into analyzing textual data and detecting shifts in identity.

## Installation

- Install the DIDit module by running the following commands:

# Installation
Clone the repository:
$ git clone https://github.com/yourusername/DIDit.git

Install the dependencies using pip:
$ pip install -r requirements.txt

Usage
Modify the app.py file to specify the desired RSS feed and any additional customization options.

Run the application using the following command:

$ python app.py

Open your web browser and visit http://localhost:8050 to access the Dash application.

Project Structure
The project directory structure is organized as follows:

app.py: The main file that contains the Dash application and the data processing logic.
visualize.py: Contains visualization functions for character frequencies, word frequencies, and stopword/non-letter frequencies.
graph.py: Contains functions for graph creation and conversion.
timeline.py: Contains functions for generating the timeline time series from the RSS feed.
README.md: Documentation file explaining the project and its usage.
requirements.txt: File listing the project dependencies.
