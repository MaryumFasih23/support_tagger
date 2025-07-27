# AI Support Ticket Tagger

## Overview

The `support_tagger` project is an advanced support ticket classification system designed to automate the tagging of support tickets using artificial intelligence. Built with Streamlit and powered by the Flan-T5 transformer model, this application is primarily driven by the `app_ui.py` file, which provides a user-friendly interface for uploading CSV files or manually entering tickets. The tool categorizes tickets with relevant tags using both zero-shot and few-shot learning approaches, enhancing support team efficiency.

## Features

* **Interactive UI (`app_ui.py`)**: A highly customizable web interface with advanced CSS styling, real-time progress indicators, and session management for processed tickets.
* **Dual Tagging Methods**: Implements zero-shot and few-shot classification to ensure robust and accurate tag predictions.
* **File Upload Support**: Processes multiple tickets from a CSV file with a `ticket_text` column.
* **Real-Time Feedback**: Displays processing status, metrics (e.g., total processed tickets), and results instantly.
* **Tag Validation**: Ensures exactly three tags are assigned per ticket from a predefined set, with fallback mechanisms for incomplete predictions.
* **Session Management**: Allows clearing results or resetting the model via sidebar controls.

## What This Project Does

This project develops an intelligent system that analyzes support ticket descriptions and assigns up to three relevant tags (e.g., "error," "website," "payment") to facilitate ticket prioritization and routing. The `app_ui.py` script offers a polished interface for uploading batch files or entering individual tickets, while processing results are displayed with both zero-shot and few-shot tag outputs, making it a valuable tool for support operations.

## What This Project Helps With

* **Ticket Management**: Streamlines the classification of support tickets, saving time for support teams.
* **Workflow Efficiency**: Enables quicker assignment of tickets to appropriate teams based on tags.
* **Scalability**: Handles large volumes of tickets via CSV uploads, ideal for busy support centers.
* **Training Data Generation**: Provides insights into ticket patterns, aiding in the development of future AI models.

## What This Project Teaches You

* **Natural Language Processing (NLP)**: Learn to apply transformer models like Flan-T5 for text classification tasks.
* **Streamlit Development**: Gain skills in creating interactive web applications with custom styling and dynamic updates.
* **Data Processing**: Understand CSV handling and text analysis for real-world applications.
* **Machine Learning Pipelines**: Explore zero-shot and few-shot learning techniques for flexible model deployment.
* **UI/UX Design**: Master integrating CSS for enhanced user interfaces in Streamlit.
* **Error Handling**: Learn to implement robust validation and fallback mechanisms in AI applications.

## Prerequisites

* Python 3.8 or higher
* pip (Python package manager)

## Dependencies

All required Python packages are listed in the `requirements.txt` file. Install them using:

```
pip install -r requirements.txt
```

**Key dependencies include:**

* `streamlit`: For building the web-based UI
* `pandas`: For CSV file processing and data manipulation
* `transformers`: For loading and using the Flan-T5 model
* `torch`: Required by transformers for model operations

### Notes on `requirements.txt`

* Contains specific package versions for compatibility
* Use a virtual environment (e.g., `venv`) to avoid conflicts with other projects
* Requires an internet connection for initial package and model downloads

## Installation

**Clone the Repository:**

```
git clone <repository-url>
cd support_tagger
```

**Set Up a Virtual Environment (Recommended):**

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install Dependencies:**
Run the command provided in the "Dependencies" section above.

**Prepare the CSV File:**

Create a CSV file (e.g., `tickets.csv`) with a `ticket_text` column containing ticket descriptions.

**Example:**

```
ticket_text
Payment page crashes when I click submit
I can't log in to my account
```

## Running the Application

Start the application with:

```
streamlit run app_ui.py
```

Access it in your browser at [http://localhost:8501](http://localhost:8501)

## Usage

**Upload a CSV File:**

* Use the "Upload Support Tickets" section to select your `tickets.csv` file.
* Click "Process CSV File" to process all tickets and view results.

**Manual Ticket Input:**

* Enter a single ticket in the "Enter ticket text" text area.
* Click "Process Single Ticket" to classify it.

**View Results:**

* Results are shown under "Processing Results" with zero-shot and few-shot tags for the last 10 tickets.

**Controls:**

* Use the sidebar to "Clear Results" or "Reset Model" as needed.

## Expected Output

* Each ticket is tagged with exactly three tags (e.g., "payment, error, website")
* Processing status and metrics are updated in real-time in the sidebar

## Customization

* **Tag Set**: Modify the `ALL_TAGS` list in `app_ui.py` to include or exclude tags
* **Styling**: Adjust the CSS in the `st.markdown()` call under "CUSTOM CSS STYLING"
* **Model**: Replace `google/flan-t5-base` in `load_model()` with another compatible model

## Troubleshooting

* **Model Loading Errors**: Ensure internet connectivity and sufficient memory. Reset the model via the sidebar if issues persist.
* **CSV Format Issues**: Verify the CSV has a `ticket_text` column. Refer to the example above.
* **Performance**: For large CSV files, processing may take time; monitor the progress bar.

## Contributing

Contributions are welcome! Fork the repository, make changes, and submit a pull request. Update `requirements.txt` for new dependencies.

## License

MIT License (see LICENSE file if applicable)

## Contact

For support or inquiries, open an issue in the repository or contact the project maintainers.

## Author

Maryum Fasih  
