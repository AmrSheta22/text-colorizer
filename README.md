# text-colorizer
 A tool that takes a set of text files as input and highlights each sentence (using HTML tags), such that the more two sentences are similar, the more their highlight colors are also similar.
 ![image](https://github.com/AmrSheta22/text-colorizer/blob/main/static/assets/logo.png)
 <br>
 This code can take docx and pdf files also and can use input text as well. 
 <br>
 It uses a sentence transformer to generate embeddings then compute the cosine distance matrix. Then use MDS to reduce the distance matrix to one dimension. Finally, use the normalized array to get the colors and use the colors to generate the HTML content.
# Usage
 To use this tool, you need to install the dependencies using the following command:
  <br>
 `pip install -r requirements.txt`
  <br>
 Then run the app directly using the following command:
  <br>
 `python src/app.py`
  <br>
 This command will start the Flask server and allow you to upload and process text files using the web interface at `127.0.0.1:5000` which is the default port.
# Screenshots
 ![image](https://github.com/AmrSheta22/text-colorizer/blob/main/screenshots/screenshot1.png)
 ![image](https://github.com/AmrSheta22/text-colorizer/blob/main/screenshots/screenshot2.png)
 ![image](https://github.com/AmrSheta22/text-colorizer/blob/main/screenshots/screenshot3.png)

