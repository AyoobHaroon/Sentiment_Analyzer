import tkinter as tk
from tkinter import messagebox, filedialog
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load dataset and train models
categories = ['rec.sport.hockey', 'sci.space', 'comp.graphics', 'talk.politics.mideast']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=42)

# Train Naive Bayes model
nb_pipeline = make_pipeline(CountVectorizer(), MultinomialNB())
nb_pipeline.fit(X_train, y_train)

# Train Logistic Regression model
lr_pipeline = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
lr_pipeline.fit(X_train, y_train)

# Save the trained models
with open('nb_model.pkl', 'wb') as model_file:
    pickle.dump(nb_pipeline, model_file)

with open('lr_model.pkl', 'wb') as model_file:
    pickle.dump(lr_pipeline, model_file)

# Load the models
with open('nb_model.pkl', 'rb') as model_file:
    nb_model = pickle.load(model_file)

with open('lr_model.pkl', 'rb') as model_file:
    lr_model = pickle.load(model_file)

# Reverse mapping for categories
category_mapping = {i: cat for i, cat in enumerate(categories)}

# GUI Application
class SentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Sentiment Analyzer")

        self.current_model = nb_model  # Default model

        # Widgets
        tk.Label(root, text="Enter Text:").pack(pady=10)
        self.text_entry = tk.Text(root, height=10, width=50)
        self.text_entry.pack(pady=10)

        self.analyze_button = tk.Button(root, text="Analyze Sentiment", command=self.analyze_sentiment)
        self.analyze_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

        tk.Label(root, text="Choose Model:").pack(pady=10)
        self.model_selection = tk.StringVar(value="Naive Bayes")
        tk.Radiobutton(root, text="Naive Bayes", variable=self.model_selection, value="Naive Bayes", command=self.switch_model).pack()
        tk.Radiobutton(root, text="Logistic Regression", variable=self.model_selection, value="Logistic Regression", command=self.switch_model).pack()

        self.visualize_button = tk.Button(root, text="Visualize Probabilities", command=self.visualize_probabilities)
        self.visualize_button.pack(pady=10)

        self.export_button = tk.Button(root, text="Export Results", command=self.export_results)
        self.export_button.pack(pady=10)

    def analyze_sentiment(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Input Error", "Please enter some text.")
            return

        prediction = self.current_model.predict([text])[0]
        category = category_mapping[prediction]

        self.result_label.config(text=f"Predicted Category: {category}")

    def switch_model(self):
        model_name = self.model_selection.get()
        if model_name == "Naive Bayes":
            self.current_model = nb_model
        elif model_name == "Logistic Regression":
            self.current_model = lr_model

    def visualize_probabilities(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Input Error", "Please enter some text.")
            return

        probabilities = self.current_model.predict_proba([text])[0]
        categories = [category_mapping[i] for i in range(len(probabilities))]

        plt.figure(figsize=(8, 5))
        plt.bar(categories, probabilities, color="skyblue")
        plt.title("Category Probabilities")
        plt.ylabel("Probability")
        plt.show()

    def export_results(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Input Error", "Please enter some text.")
            return

        prediction = self.current_model.predict([text])[0]
        category = category_mapping[prediction]

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "w") as file:
                file.write(f"Input Text:\n{text}\n\n")
                file.write(f"Predicted Category: {category}\n")
            messagebox.showinfo("Success", "Results exported successfully!")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    root.mainloop()
