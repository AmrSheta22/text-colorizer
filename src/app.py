from flask import Flask, request, jsonify, send_from_directory
import os
import docx
import PyPDF2
import numpy as np
import re
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_folder='../static', static_url_path='')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def compute_sentence_embeddings(sentences):
    return model.encode(sentences)

def compute_cosine_distance_matrix(embeddings):
    return squareform(pdist(embeddings, 'cosine'))

def reduce_to_one_dimension(distance_matrix):
    """
    Reduces a distance matrix to one dimension using Multidimensional Scaling (MDS).

    Parameters:
    distance_matrix (numpy.ndarray): The input distance matrix.

    Returns:
    numpy.ndarray: A one-dimensional representation of the input distance matrix.
    """
    mds = MDS(n_components=1, dissimilarity='precomputed', random_state=42)
    one_dimensional = mds.fit_transform(distance_matrix).flatten()
    return one_dimensional

def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

def get_color(value):
    """
    Given a value between 0 and 1, this function returns an RGB color value that transitions
    smoothly between red, yellow, green, cyan, and blue. The value parameter determines the 
    position of the color in the transition. The color is calculated by interpolating between 
    the colors in the transitions list. The function ensures that the value is within the 
    range [0, 1] and calculates the segment and local value based on the segment length. If 
    the segment is equal to the number of segments, the segment is set to the last segment and 
    the local value is set to 1. The function then calculates the interpolated color by 
    interpolating between the start and end colors in the transitions list. The function 
    returns the interpolated color as a tuple of three integers representing the RGB values.
    
    :param value: A float between 0 and 1 that determines the position of the color in the 
                  transition.
    :type value: float
    :return: An RGB color value as a tuple of three integers representing the RGB values.
    :rtype: tuple(int, int, int)
    """
    transitions = [
        (255, 0, 0),   # Red
        (255, 255, 0), # Yellow
        (0, 255, 0),   # Green
        (0, 255, 255), # Cyan
        (0, 0, 255),    # Blue
    ]
    value = max(0, min(1, value))
    num_segments = len(transitions) - 1
    segment_length = 1.0 / num_segments
    segment = int(value / segment_length)
    if segment == num_segments:
        segment = num_segments - 1
        local_value = 1.0
    else:
        local_value = (value - segment * segment_length) / segment_length
    start_color = transitions[segment]
    end_color = transitions[segment + 1]
    interpolated_color = tuple(
        int(start_color[i] + local_value * (end_color[i] - start_color[i]))
        for i in range(3)
    )
    return interpolated_color

def get_color_map(normalized_array):
    colors = [get_color(i) for i in normalized_array]
    return colors

def generate_html(sentences, colors):
    """
    Generates an HTML content based on the input sentences and their corresponding colors.

    Parameters:
    sentences (list): A list of sentences to be included in the HTML.
    colors (list): A list of colors corresponding to each sentence.

    Returns:
    str: The generated HTML content.
    """
    html_content = '''
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                background-color: #f5f5f5;
                padding: 20px;
            }
            .sentence {
                display: inline-block;
                margin-bottom: 10px;
                padding: 5px;
                border-radius: 5px;
                white-space: pre-wrap; /* Preserve spaces and newlines */
            }
            .file-separator {
                display: block;
                margin: 20px 0;
                height: 1px;
                background-color: #ccc;
            }
        </style>
    </head>
    <body>
    '''
    for i, sentence in enumerate(sentences):
        color = colors[i]
        html_content += f'<span class="sentence" style="background-color: rgb{color}">{sentence}</span> '
        if '\n' in sentence:
            html_content += '<br>'

    html_content += '</body></html>'
    return html_content

def get_cluster_labels(embeddings):
    ss = StandardScaler()
    scaled_embeds = ss.fit_transform(embeddings)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=32).fit(scaled_embeds)
    return clustering.labels_

def average_for_each_cluster(embeddings, labels):
    clusters = {}
    for i, j in zip(labels, embeddings):
        if i not in clusters:
            clusters[i] = []
        clusters[i].append(j)
    for i in clusters:
        clusters[i] = sum(clusters[i]) / len(clusters[i])
    clusters = dict(sorted(clusters.items()))
    return clusters

def index_mapping(lst):
    index_dict = {}
    for index, value in enumerate(lst):
        if value in index_dict:
            index_dict[value].append(index)
        else:
            index_dict[value] = [index]
    index_dict = dict(sorted(index_dict.items()))
    return index_dict

def get_one_dimentional_clusters(index_dict, embeddings):
    one_dimensional_clusters = {}
    for i, k in index_dict.items():
        this_cluster = []
        if len(k) == 1:
            one_dimensional_clusters[i] = [0.5]
        else:
            for j in k:
                this_cluster.append(embeddings[j])
            cosine_distance_matrix = compute_cosine_distance_matrix(np.array(this_cluster))
            one_dimensional_cluster = reduce_to_one_dimension(cosine_distance_matrix)
            normalized_array = normalize_array(one_dimensional_cluster)
            one_dimensional_clusters[i] = normalized_array
    one_dimensional_clusters = dict(sorted(one_dimensional_clusters.items()))
    return one_dimensional_clusters

def extract_colors(cluster_values, index_dict, cluster_colors):
    colored_points = {}
    for cluster, values in cluster_values.items():
        max_color = cluster_colors[cluster]
        colored_points[cluster] = [interpolate_color(value, max_color) for value in values]
    colors_index = {}
    for k1, k2 in zip(index_dict.values(), colored_points.values()):
        for i, j in zip(k1, k2):
            colors_index[i] = tuple(j)
    colors = list(dict(sorted(colors_index.items())).values())
    return colors

def interpolate_color(value, max_color):
    return np.clip(np.array(max_color) - int(value*20), 0, 255)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

import PyPDF2

def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    texts = data.get('texts', [])
    all_sentences = [sentence for text in texts for sentence in split_into_sentences(text)]
    embeddings = compute_sentence_embeddings(all_sentences)
    cosine_distance_matrix = compute_cosine_distance_matrix(embeddings)
    one_dimensional = reduce_to_one_dimension(cosine_distance_matrix)
    normalized_array = normalize_array(one_dimensional)
    colors = get_color_map(normalized_array)
    html_output = generate_html(all_sentences, colors)
    return jsonify({'html': html_output})

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    all_texts = []
    for file in files:
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        ext = filename.split('.').pop().lower()
        text = ''

        if ext == 'docx':
            try:
                text = extract_text_from_docx(file_path)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        elif ext == 'pdf':
            try:
                text = extract_text_from_pdf(file_path)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        os.remove(file_path)
        all_texts.append(text)
    text_input = request.form.get("textInput")
    all_texts.append(text_input)
    if all_texts == []:
        return jsonify({'error': 'No files uploaded and to text was input'}), 400
    if all_texts == ['']:
        return jsonify({'error': 'No text input'}), 400
    # Add a separator between different files
    combined_text = '\n\n---\n\n'.join(all_texts)
    all_sentences = split_into_sentences(combined_text)
    if len(all_sentences) == 1:
        return jsonify({'error ': 'no other text input to comapre to'}), 400
    embeddings = compute_sentence_embeddings(all_sentences)
    labels = get_cluster_labels(embeddings)
    clusters = average_for_each_cluster(embeddings, labels)
    index_dict = index_mapping(labels)
    cluster_values = get_one_dimentional_clusters(index_dict, embeddings)
    cluster_embeddings = list(clusters.values())
    cosine_distance_matrix = compute_cosine_distance_matrix(cluster_embeddings)
    one_dimensional = reduce_to_one_dimension(cosine_distance_matrix)
    normalized_array = normalize_array(one_dimensional)
    cluster_colors = dict(enumerate(get_color_map(normalized_array)))
    colors = extract_colors(cluster_values, index_dict, cluster_colors)
    html_output = generate_html(all_sentences, colors)
    return jsonify({'html': html_output})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
