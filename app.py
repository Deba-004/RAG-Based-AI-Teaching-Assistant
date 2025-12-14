from flask import Flask, render_template, request, redirect, url_for
from process_query import merged_func

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    question = None

    if request.method == "POST":
        question = request.form["query"]
        answer = merged_func(question)

    return render_template("index.html", question=question, answer=answer)

@app.route("/ask-again")
def ask_again():
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)