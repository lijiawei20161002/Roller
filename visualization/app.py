from flask import Flask, render_template, request, jsonify, Response, redirect, url_for  
  
app = Flask(__name__)  

@app.route("/")
def home():
    return render_template("index.html")
  
@app.route("/submit", methods=["POST"])  
def submit(): 
    serverNumber = request.form["serverNumber"]
    #print(serverNumber)
    return redirect(url_for('home')) 
 
if __name__ == "__main__":  
    app.run(debug=True, port=5001)  
