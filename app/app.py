import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide warning with tensorflow
from flask import Flask, render_template, request, jsonify
from keras.preprocessing import image

from pyimagesearch.rmac import describe, Searcher
import cv2
import pickle


# create flask instance
app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))



# main route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    if request.method == "POST":

        RESULTS_ARRAY = []

        # get url

        image_url = request.form.get('img')
        with open('app/index', 'rb') as handle:
            index = pickle.loads(handle.read())

        try:

            # load the query image and describe it
            img = image.load_img(UPLOAD_FOLDER+image_url)

            #compute the matrix
            descriptor = describe(img)

            #compute the results
            results = Searcher(index, descriptor)

            # loop over the results, displaying the score and image name
            for (score, resultID) in results:
                RESULTS_ARRAY.append(
                    {"image": str(resultID), "score": str(score)})

            # return success
            return jsonify(results=(RESULTS_ARRAY[::-1][:3]))

        except:

            # return error
            return jsonify({"sorry": "Sorry, no results! Please try again."}), 500


# run!
if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
