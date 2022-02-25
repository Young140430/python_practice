from flask import Flask
from flask import request
import io
from PIL import Image
app = Flask(__name__)


@app.route("/xxx",methods=["POST"])
def xxx():
    print(request.form.get("name"))

    file = request.files.get("file")
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))
    image.show()
    return "xxx!"

if __name__ == '__main__':
    app.run(port=5000)

