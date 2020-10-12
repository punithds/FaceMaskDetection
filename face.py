import os
import tagger
from PIL import Image
from flask import Flask,render_template,request,send_from_directory
app=Flask(__name__)

# APP_ROOT=os.path.dirname(os.path.abspath(__file__))

@app.route("/")
@app.route("/home",methods=["GET","POST"])

def home():
	
	return render_template("index.html")


@app.route("/upload",methods=["GET","POST"])
def upload():
    	
	target="./static/predicted"
	# print(os.listdir(target))
	
	for uploaded_file in request.files.getlist('file'):
		print(os.path.join(target,uploaded_file.filename))
		uploaded_file.save(os.path.join(target,uploaded_file.filename))
		print("done")
		filepath=os.path.join(target,uploaded_file.filename)
		print(filepath)
		tag=tagger.TagImages(filepath)
		tag.draw_box_predicted()

		name=os.path.join("./static/images",uploaded_file.filename)
	
	return render_template('index.html', name=name)

if __name__ == '__main__':
	app.run(port=4956,debug=True)

# import os
# # print(os.getcwd())
# # print(os.path.join(os.getcwd(),"static\predicted\download.jpg"))
# print(os.getcwd())