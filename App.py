from flask import Flask, request, render_template_string
import webbrowser
import threading
import pandas as pd
from Final.text_statistics import show_text_statistics
from Final.readability_analysis import LexicalFeatures, VocabularyRichnessFeatures, ReadabilityScores
import araana
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("arabic-ner")
model = AutoModelForTokenClassification.from_pretrained("arabic-ner")

# Update the tokenizer with the max_length parameter
tokenizer.model_max_length = 10000000

# Initialize NER pipeline
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)

# Initialize AraNet models
sentiment_obj = araana.AraAna('araana/models/sentiment_araana')
dialect_obj = araana.AraAna('araana/models/dialect_araana')
emotion_obj = araana.AraAna('araana/models/emotion_araana')
irony_obj = araana.AraAna('araana/models/irony_araana')

app = Flask(__name__)

@app.route('/')
def home():
    sample_text = "و نقلت وكالة رويترز عن ثلاثة دبلوماسيين في الاتحاد الأوروبي ، أن بلجيكا و إيرلندا و لوكسمبورغ تريد أيضاً مناقشة"
    return f"""
    <html dir="rtl">
        <head>
            <style>
                body {{
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #d4af37, #2e8b57);
                    font-family: Arial, sans-serif;
                    color: #fff;
                }}
                .container {{
                    text-align: center;
                    background-color: rgba(0, 0, 0, 0.6);
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                }}
                h1 {{
                    color: #ffd700;
                }}
                textarea {{
                    width: 100%;
                    max-width: 500px;
                    height: 150px;
                    border: 2px solid #ffd700;
                    border-radius: 5px;
                    padding: 10px;
                    box-sizing: border-box;
                    font-size: 16px;
                    color: #333;
                    text-align: right;
                }}
                input[type="submit"], input[type="file"], .features-button {{
                    background-color: #ffd700;
                    border: none;
                    color: #333;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 10px 0;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }}
                input[type="submit"]:hover, input[type="file"]:hover, .features-button:hover {{
                    background-color: #ffb700;
                }}
                .footer {{
                    position: absolute;
                    bottom: 10px;
                    text-align: center;
                    width: 100%;
                }}
                .team {{
                    margin-top: 20px;
                    text-align: center;
                }}
                .team h2 {{
                    color: #ffd700;
                    text-transform: uppercase;
                }}
                .team-grid {{
                    display: flex;
                    justify-content: center;
                    gap: 20px;
                    flex-wrap: wrap;
                }}
                .team-member {{
                    text-align: center;
                }}
                .team-member img {{
                    border-radius: 50%;
                    width: 100px;
                    height: 100px;
                }}
                .team-member a {{
                    color: #ffd700;
                    text-decoration: none;
                }}
                .team-member a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Arabic Text Analysis</h1>
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <textarea name="text">{sample_text}</textarea><br>
                    <input type="file" name="file"><br>
                    <input type="submit" value="Analyze">
                </form>
            </div>
            <div class="team">
                <h2>Team</h2>
                <div class="team-grid">
                    <div class="team-member">
                        <img src="https://github.com/mohmmedelfateh.png" alt="Mohamed Elfateh">
                        <p><a href="https://github.com/mohmmedelfateh" target="_blank">Mohamed Elfateh</a></p>
                    </div>
                    <div class="team-member">
                        <img src="https://github.com/mostafa4nabih.png" alt="Mostafa Ahmed">
                        <p><a href="https://github.com/mostafa4nabih" target="_blank">Mostafa Ahmed</a></p>
                    </div>
                    <div class="team-member">
                        <img src="https://github.com/mohammedhisham1.png" alt="Mohamed Hesham">
                        <p><a href="https://github.com/mohammedhisham1" target="_blank">Mohamed Hesham</a></p>
                    </div>
                    <div class="team-member">
                        <img src="https://github.com/sexynade.png" alt="Ahmed Soltan">
                        <p><a href="https://github.com/sexynade" target="_blank">Ahmed Soltan</a></p>
                    </div>
                </div>
            </div>
            <br><br>
            <div class="footer">
                <p>All rights Reserved 2023-2024 El-Detector</p>
            </div>
        </body>
    </html>
    """

@app.route('/features')
def features():
    return """
    <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }
                h1, h2 {
                    color: #2e8b57;
                }
                ul {
                    list-style-type: none;
                    padding: 0;
                }
                li {
                    margin-bottom: 10px;
                }
                .feature {
                    background: #f9f9f9;
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 5px;
                }
                .back-link {
                    color: #2e8b57;
                    text-decoration: none;
                    font-size: 16px;
                    margin-top: 20px;
                    display: inline-block;
                }
                .back-link:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Features Description</h1>

            <h2>Lexical Features Summary</h2>
            <ul>
                <li class="feature">
                    <strong>Average Word Length</strong><br>
                    <em>Description:</em> Calculates the average length of words in a text.<br>
                    <em>Purpose:</em> Provides insight into the complexity of the words used in the text.
                </li>
                <li class="feature">
                    <strong>Average Sentence Length By Word</strong><br>
                    <em>Description:</em> Calculates the average number of words per sentence.<br>
                    <em>Purpose:</em> Helps measure the complexity and readability of the text by indicating sentence length.
                </li>
                <li class="feature">
                    <strong>Average Sentence Length By Character</strong><br>
                    <em>Description:</em> Calculates the average number of characters per sentence.<br>
                    <em>Purpose:</em> Offers an alternative measure of sentence complexity, focusing on character count rather than word count.
                </li>
                <li class="feature">
                    <strong>Special Character Count</strong><br>
                    <em>Description:</em> Counts the number of non-alphanumeric characters (e.g., punctuation, symbols) in the text.<br>
                    <em>Purpose:</em> Indicates the presence and frequency of special characters, which can affect readability and style.
                </li>
                <li class="feature">
                    <strong>Average Syllable per Word</strong><br>
                    <em>Description:</em> Calculates the average number of syllables per word.<br>
                    <em>Purpose:</em> Provides a measure of word complexity, with higher averages indicating more complex or longer words.
                </li>
                <li class="feature">
                    <strong>Functional Words Count</strong><br>
                    <em>Description:</em> Counts the number of common functional words (e.g., prepositions, conjunctions) in the text.<br>
                    <em>Purpose:</em> Helps identify the use of essential but less content-rich words, which can influence the text's structure and flow.
                </li>
                <li class="feature">
                    <strong>Punctuation Count</strong><br>
                    <em>Description:</em> Counts the number of punctuation marks in the text.<br>
                    <em>Purpose:</em> Indicates the use of punctuation, which affects the text's readability, pauses, and overall structure.
                </li>
            </ul>

            <h2>Vocabulary Richness Features</h2>
            <ul>
                <li class="feature">
                    <strong>Hapax Legomenon</strong><br>
                    <em>Description:</em> The number of words that occur exactly once in the text.<br>
                    <em>Purpose:</em> Measures lexical diversity by counting unique words.
                </li>
                <li class="feature">
                    <strong>Hapax DisLegemena</strong><br>
                    <em>Description:</em> The number of words that occur exactly twice in the text.<br>
                    <em>Purpose:</em> Another measure of lexical diversity, focusing on words with low frequency.
                </li>
                <li class="feature">
                    <strong>Honores R Measure</strong><br>
                    <em>Description:</em> The ratio of hapax legomena to the total number of words.<br>
                    <em>Purpose:</em> Indicates the proportion of unique words in the text.
                </li>
                <li class="feature">
                    <strong>Sichel’s Measure</strong><br>
                    <em>Description:</em> The ratio of hapax dislegemena to the total number of words.<br>
                    <em>Purpose:</em> Provides insight into the occurrence of words that appear infrequently.
                </li>
                <li class="feature">
                    <strong>Brunet’s Measure W</strong><br>
                    <em>Description:</em> A measure that combines the number of unique words and the total number of words.<br>
                    <em>Purpose:</em> Provides a sophisticated measure of lexical richness.
                </li>
                <li class="feature">
                    <strong>Yule’s Characteristic K</strong><br>
                    <em>Description:</em> A measure based on the frequency distribution of words.<br>
                    <em>Purpose:</em> Indicates the concentration of word frequencies in the text.
                </li>
                <li class="feature">
                    <strong>Shannon Entropy</strong><br>
                    <em>Description:</em> A measure of the unpredictability or randomness in the word distribution.<br>
                    <em>Purpose:</em> Reflects the complexity and diversity of the vocabulary.
                </li>
                <li class="feature">
                    <strong>Simpson’s Index</strong><br>
                    <em>Description:</em> A measure of the probability that two randomly selected words are the same.<br>
                    <em>Purpose:</em> Indicates the dominance of certain words in the text.
                </li>
            </ul>

            <h2>Readability Scores</h2>
            <ul>
                <li class="feature">
                    <strong>Flesch Reading Ease</strong><br>
                    <em>Description:</em> Calculates readability based on sentence length and syllable count.<br>
                    <em>Purpose:</em> Higher scores indicate easier readability.
                </li>
                <li class="feature">
                    <strong>Flesch-Kincaid Grade Level</strong><br>
                    <em>Description:</em> Calculates readability in terms of the U.S. school grade level.<br>
                    <em>Purpose:</em> Indicates the education level required to understand the text.
                </li>
                <li class="feature">
                    <strong>Gunning Fog Index</strong><br>
                    <em>Description:</em> Estimates readability based on sentence length and complex word count.<br>
                    <em>Purpose:</em> Higher scores indicate more difficult texts.
                </li>
                <li class="feature">
                    <strong>Dale Chall Readability Formula</strong><br>
                    <em>Description:</em> Measures readability based on sentence length and percentage of difficult words.<br>
                    <em>Purpose:</em> Lower scores indicate easier readability.
                </li>
                <li class="feature">
                    <strong>Shannon Entropy</strong><br>
                    <em>Description:</em> Measures the unpredictability or randomness of the word distribution.<br>
                    <em>Purpose:</em> Reflects the complexity and diversity of the vocabulary.
                </li>
                <li class="feature">
                    <strong>Simpson’s Index</strong><br>
                    <em>Description:</em> Measures the probability that two randomly selected words are the same.<br>
                    <em>Purpose:</em> Indicates the dominance of certain words in the text.
                </li>
            </ul>
            <a href="/" class="back-link">Back to Home</a>
        </body>
    </html>
    """

@app.route('/analyze', methods=['POST'])
def analyze_text():
    text = request.form['text']
    file = request.files.get('file')

    if file:
        text = file.read().decode('utf-8')

    if len(text) > 500:
        text = text[:500]


    # Text statistics
    statistics = show_text_statistics(pd.Series([text]))
    statistics_html = statistics.to_html(classes='center-table', index=False)

    # Readability analysis
    lexical_features = LexicalFeatures([text])
    vocabulary_richness = VocabularyRichnessFeatures([text])
    readability_scores = ReadabilityScores([text])

    lexical_features_df = lexical_features.compute_features()
    vocabulary_richness_df = vocabulary_richness.compute_features()
    readability_scores_df = readability_scores.compute_features()

    lexical_features_html = lexical_features_df.to_html(classes='center-table', index=False)
    vocabulary_richness_html = vocabulary_richness_df.to_html(classes='center-table', index=False)
    readability_scores_html = readability_scores_df.to_html(classes='center-table', index=False)

    # Sentiment and intent analysis
    sentiment_result = 'Positive Text' if sentiment_obj.predict(text=text)[0][0] == 'pos' else 'Negative Text'
    dialect_result = dialect_obj.predict(text=text)[0][0].replace("_", " ")
    emotion_result = emotion_obj.predict(text=text)[0][0].title()
    irony_result = 'Sarcasm Text' if irony_obj.predict(text=text)[0][0] == '1' else 'Antisarcasm Text'

    sentiment_html = f"""
    <p><strong>Sentiment Analysis:</strong> {sentiment_result}</p>
    <p><strong>Dialect Analysis:</strong> {dialect_result}</p>
    <p><strong>Emotion Analysis:</strong> {emotion_result}</p>
    <p><strong>Irony Analysis:</strong> {irony_result}</p>
    """

    # Named Entity Recognition (NER)
    ner_results = ner_pipeline(text)
    ner_html = """
    <table class="center-table">
        <tr>
    
            <th>Word</th>
            <th>Entity</th>
        </tr>
    """
    for entity in ner_results:
        ner_html += f"<tr><td>{entity['word']}</td><td>{entity['entity']}</td></tr>"
    ner_html += "</table>"

    return render_template_string(f"""
    <html dir="rtl">
        <head>
            <style>
                body {{
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #d4af37, #2e8b57);
                    font-family: Arial, sans-serif;
                    color: #fff;
                }}
                .container {{
                    text-align: center;
                    background-color: rgba(0, 0, 0, 0.6);
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                }}
                .center-table {{
                    margin-left: auto;
                    margin-right: auto;
                    border-collapse: collapse;
                    text-align: center;
                    }}
                    .center-table th, .center-table td {{
                    border: 1px solid black;
                    padding: 8px;
                    text-align: center;
                    }}

                table {{
                    width: 100%;
                    max-width: 800px;
                    margin-top: 20px;
                    border-collapse: collapse;
                    text-align: center;
                    color: #fff;
                }}
                th, td {{
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                    color: #333;
                }}
                tr:hover {{
                    background-color: #ddd;
                    color: #333;
                }}
                .footer {{
                    text-align: center;
                    width: 100%;
                    margin-top: 20px;
                }}
                .back-link {{
                    color: #ffd700;
                    text-decoration: none;
                }}
                .back-link:hover {{
                    text-decoration: underline;
                }}
                .features-button {{
                    background-color: #ffd700;
                    border: none;
                    color: #333;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 10px 0;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }}
                .features-button:hover {{
                    background-color: #ffb700;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Text Analysis Results</h1>
                <h2>Text Statistics</h2>
                {statistics_html}
                <h2>Lexical Features</h2>
                {lexical_features_html}
                <h2>Vocabulary Richness</h2>
                {vocabulary_richness_html}
                <h2>Readability Scores</h2>
                {readability_scores_html}
                <h2>Sentiment Analysis and Intent Analysis</h2>
                {sentiment_html}
                <h2>Named Entity Recognition (NER)</h2>
                {ner_html}
                <div class="footer">
                    <a href="/" class="back-link">Back to Home</a>
                    <br>
                    <a href="/features" class="features-button">Features Description</a>
                </div>
            </div>
        </body>
    </html>
    """)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Start the Flask app in a separate thread
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
