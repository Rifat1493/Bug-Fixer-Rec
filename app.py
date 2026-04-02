import os
import pickle
import re
import json
import zipfile
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import nltk
from gensim.models import FastText
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

TRAIN_CSV_PATH = "data/train.csv"
MODEL_PATH = "data/best_cnn_lstm_supcon_classifier.keras"
TFIDF_PATH = "data/tfidf_vectorizer.pkl"
FASTTEXT_PATH = "data/fasttext.pkl"

MAX_WORDS = 50
VEC_DIM = 100
TOP_K_TFIDF = 50
MIN_WORDS = 5
SHAP_BACKGROUND_SIZE = 32
TOP_K_FEATURES = 5


def preprocess_text(texts, stop_words, lemmatizer):
	cleaned = []
	for text in texts:
		text = text.lower()
		text = re.sub(r"http\S+|www\S+", "", text)
		text = re.sub(r"[^a-zA-Z.\-\s]", "", text)
		text = re.sub(r"\s+", " ", text).strip()
		words = []
		for w in re.split(r"[\s.-]+", text):
			if not w:
				continue
			if len(w) > 15:
				continue
			if w in stop_words:
				continue
			words.append(lemmatizer.lemmatize(w))
		cleaned.append(" ".join(words))
	return cleaned


def filter_by_min_words(texts, labels, min_words=5):
	filtered_texts = []
	filtered_labels = []
	for text, label in zip(texts, labels):
		if len(text.split()) >= min_words:
			filtered_texts.append(text)
			filtered_labels.append(label)
	return filtered_texts, filtered_labels


def select_top_k_tfidf(texts, vectorizer, top_k):
	tfidf = vectorizer.transform(texts)
	feature_names = vectorizer.get_feature_names_out()
	top_texts = []
	for row_idx in range(tfidf.shape[0]):
		row = tfidf.getrow(row_idx)
		if row.nnz == 0:
			top_texts.append(texts[row_idx])
			continue
		data = row.data
		indices = row.indices
		if row.nnz > top_k:
			top_idx = data.argsort()[-top_k:]
			top_terms = {feature_names[indices[i]] for i in top_idx}
		else:
			top_terms = {feature_names[i] for i in indices}
		tokens = texts[row_idx].split()
		filtered = [t for t in tokens if t in top_terms]
		top_texts.append(" ".join(filtered) if filtered else texts[row_idx])
	return top_texts


def tokenize_texts(texts, max_words):
	tokenized = []
	for text in texts:
		tokens = text.split()
		tokenized.append(tokens[:max_words])
	return tokenized


def get_ft_embedding_single(tokens, ft_model, max_words, vec_dim):
	embedding = np.zeros((max_words, vec_dim), dtype=np.float32)
	for j, token in enumerate(tokens[:max_words]):
		if token in ft_model.wv:
			embedding[j, :] = ft_model.wv[token]
	return embedding


def build_background_embeddings(texts, ft_model, max_words, vec_dim, max_rows):
	if not texts:
		return np.zeros((1, max_words, vec_dim), dtype=np.float32)
	max_rows = min(max_rows, len(texts))
	rng = np.random.RandomState(42)
	indices = rng.choice(len(texts), size=max_rows, replace=False)
	background = np.zeros((max_rows, max_words, vec_dim), dtype=np.float32)
	for i, idx in enumerate(indices):
		tokens = texts[idx].split()[:max_words]
		background[i, :, :] = get_ft_embedding_single(tokens, ft_model, max_words, vec_dim)
	return background


def init_shap_explainer(model, background):
	try:
		import shap
		return shap.GradientExplainer(model, background), None
	except Exception:
		try:
			import shap
			return shap.DeepExplainer(model, background), None
		except Exception as exc:
			return None, f"SHAP unavailable: {exc}"


def summarize_shap_tokens(shap_values, tokens, top_k):
	if not tokens:
		return [], []
	if shap_values.ndim == 4:
		shap_values = shap_values[0]
	if shap_values.ndim != 3:
		return [], []
	max_idx = min(len(tokens), shap_values.shape[1])
	scores = []
	for i in range(max_idx):
		score = float(np.sum(np.abs(shap_values[0, i, :])))
		scores.append(score)
	if not scores:
		return [], []
	order = np.argsort(scores)[-top_k:][::-1]
	labels = [tokens[i] for i in order]
	values = [scores[i] for i in order]
	return labels, values


def patch_lambda_output_shape(model_path, output_path):
	with tempfile.TemporaryDirectory() as tmp_dir:
		with zipfile.ZipFile(model_path, "r") as zip_file:
			zip_file.extractall(tmp_dir)
		config_path = os.path.join(tmp_dir, "config.json")
		with open(config_path, "r", encoding="utf-8") as file:
			config = json.load(file)

		layers = config.get("config", {}).get("layers", [])
		proj_units = 128
		for layer in layers:
			if layer.get("class_name") == "Dense" and layer.get("config", {}).get("name") == "proj":
				proj_units = int(layer.get("config", {}).get("units", proj_units))
				break

		for layer in layers:
			if layer.get("class_name") == "Lambda" and layer.get("config", {}).get("name") == "proj_norm":
				layer.get("config", {})["output_shape"] = [proj_units]

		with open(config_path, "w", encoding="utf-8") as file:
			json.dump(config, file)

		with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_out:
			for root, _, files in os.walk(tmp_dir):
				for name in files:
					full_path = os.path.join(root, name)
					rel_path = os.path.relpath(full_path, tmp_dir)
					zip_out.write(full_path, rel_path)


def patch_loaded_lambda_layers(model):
	def proj_norm_fn(tensor):
		return tf.math.l2_normalize(tensor, axis=1)

	for layer in model.layers:
		if layer.__class__.__name__ == "Lambda" and layer.name == "proj_norm":
			layer.function = proj_norm_fn
			if hasattr(layer, "_function"):
				layer._function = proj_norm_fn
			if hasattr(layer, "_fn"):
				layer._fn = proj_norm_fn
	return model


def load_model_safely(model_path):
	custom_objects = {"tf": tf}
	if hasattr(tf.keras, "config") and hasattr(tf.keras.config, "enable_unsafe_deserialization"):
		tf.keras.config.enable_unsafe_deserialization()
	try:
		model = tf.keras.models.load_model(
			model_path,
			safe_mode=False,
			custom_objects=custom_objects,
		)
		return patch_loaded_lambda_layers(model)
	except TypeError:
		try:
			model = tf.keras.models.load_model(
				model_path,
				compile=False,
				custom_objects=custom_objects,
			)
			return patch_loaded_lambda_layers(model)
		except TypeError:
			return patch_loaded_lambda_layers(tf.keras.models.load_model(
				model_path,
				custom_objects=custom_objects,
			))
	except NotImplementedError as exc:
		if "output_shape" not in str(exc):
			raise
		patched_path = model_path.replace(".keras", "_patched.keras")
		patch_lambda_output_shape(model_path, patched_path)
		return patch_loaded_lambda_layers(tf.keras.models.load_model(
			patched_path,
			safe_mode=False,
			custom_objects=custom_objects,
		))


def plot_donut(labels, values, title):
	if not labels or not values:
		st.info("No features available for the donut chart.")
		return
	fig, ax = plt.subplots(figsize=(2.4, 1.5))
	wedges, _ = ax.pie(
		values,
		labels=None,
		startangle=90,
		wedgeprops={"width": 0.45, "edgecolor": "white"},
	)
	for wedge, label in zip(wedges, labels):
		angle = (wedge.theta2 + wedge.theta1) / 2.0
		x = 0.65 * np.cos(np.deg2rad(angle))
		y = 0.65 * np.sin(np.deg2rad(angle))
		ax.text(x, y, label, ha="center", va="center", fontsize=6)
	ax.set_title(title, fontsize=7, pad=6)
	ax.axis("equal")
	fig.tight_layout(pad=0.2)
	st.pyplot(fig, use_container_width=True)


def plot_horizontal_bars(labels, values, title):
	fig, ax = plt.subplots(figsize=(2.4, 1.5))
	positions = np.arange(len(labels))
	bars = ax.barh(positions, values, color="#2E86AB")
	ax.set_yticks(positions, labels)
	ax.invert_yaxis()
	ax.set_xlabel("Probability", fontsize=3)
	ax.set_title(title, fontsize=7, pad=6)
	for bar, label in zip(bars, labels):
		ax.text(
			bar.get_width() * 0.02,
			bar.get_y() + bar.get_height() / 2,
			label,
			ha="left",
			va="center",
			color="black",
			fontsize=6,
		)
	ax.set_yticklabels([])
	ax.tick_params(axis="x", labelsize=4) 
	fig.tight_layout(pad=0.2)
	st.pyplot(fig, use_container_width=True)


@st.cache_resource(show_spinner=False)
def load_resources():
	nltk.download("stopwords", quiet=True)
	nltk.download("wordnet", quiet=True)
	nltk.download("omw-1.4", quiet=True)

	stop_words = set(stopwords.words("english"))
	lemmatizer = WordNetLemmatizer()

	df_train = pd.read_csv(TRAIN_CSV_PATH)
	X_train_raw = df_train["text"].astype(str).tolist()
	y_train_raw = df_train["assigned_to"].tolist()

	X_train_clean = preprocess_text(X_train_raw, stop_words, lemmatizer)
	X_train_clean, y_train_raw = filter_by_min_words(
		X_train_clean, y_train_raw, min_words=MIN_WORDS
	)

	if not os.path.exists(TFIDF_PATH):
		raise FileNotFoundError(TFIDF_PATH)
	with open(TFIDF_PATH, "rb") as file:
		tfidf_vectorizer = pickle.load(file)

	X_train_clean = select_top_k_tfidf(X_train_clean, tfidf_vectorizer, TOP_K_TFIDF)

	if not os.path.exists(FASTTEXT_PATH):
		raise FileNotFoundError(FASTTEXT_PATH)
	with open(FASTTEXT_PATH, "rb") as file:
		ft_model = pickle.load(file)

	label_encoder = LabelEncoder()
	label_encoder.fit(y_train_raw)

	base_model = load_model_safely(MODEL_PATH)
	input_tensor = tf.keras.Input(shape=(MAX_WORDS, VEC_DIM), name="input_embeddings")
	output_tensor = base_model({"input_embeddings": input_tensor})
	model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor, name="wrapped_model")
	background = build_background_embeddings(
		X_train_clean,
		ft_model,
		MAX_WORDS,
		VEC_DIM,
		SHAP_BACKGROUND_SIZE,
	)
	shap_explainer, shap_error = init_shap_explainer(model, background)

	return {
		"stop_words": stop_words,
		"lemmatizer": lemmatizer,
		"tfidf_vectorizer": tfidf_vectorizer,
		"ft_model": ft_model,
		"label_encoder": label_encoder,
		"model": model,
		"shap_explainer": shap_explainer,
		"shap_error": shap_error,
	}


def run_inference(text, resources):
	stop_words = resources["stop_words"]
	lemmatizer = resources["lemmatizer"]
	tfidf_vectorizer = resources["tfidf_vectorizer"]
	ft_model = resources["ft_model"]
	label_encoder = resources["label_encoder"]
	model = resources["model"]
	shap_explainer = resources["shap_explainer"]

	cleaned_text = preprocess_text([text], stop_words, lemmatizer)[0]
	tfidf_text = select_top_k_tfidf([cleaned_text], tfidf_vectorizer, TOP_K_TFIDF)[0]
	tokens = tokenize_texts([tfidf_text], MAX_WORDS)[0]
	embedding = get_ft_embedding_single(tokens, ft_model, MAX_WORDS, VEC_DIM)

	model_input = np.expand_dims(embedding, axis=0)
	probs = model.predict(model_input, verbose=0)[0]
	top_indices = np.argsort(probs)[-5:][::-1]
	top_labels = label_encoder.inverse_transform(top_indices)
	top_scores = probs[top_indices]

	shap_labels = []
	shap_scores = []
	shap_error = None
	if shap_explainer is not None:
		try:
			shap_values = shap_explainer.shap_values(model_input)
			if isinstance(shap_values, list):
				class_idx = int(top_indices[0])
				class_idx = min(class_idx, len(shap_values) - 1)
				shap_values = shap_values[class_idx]
			shap_labels, shap_scores = summarize_shap_tokens(
				shap_values,
				tokens,
				TOP_K_FEATURES,
			)
		except Exception as exc:
			shap_labels, shap_scores = [], []
			shap_error = str(exc)

	return {
		"cleaned_text": cleaned_text,
		"tfidf_text": tfidf_text,
		"tokens": tokens,
		"top_labels": top_labels,
		"top_scores": top_scores,
		"shap_labels": shap_labels,
		"shap_scores": shap_scores,
		"shap_error": shap_error,
	}


st.set_page_config(
	page_title="Bug Fixer Recommendation",
	page_icon="tools",
	layout="wide",
)

# st.markdown(
# 	"""
# 	<style>
# 	.block-container { padding-top: 0.9rem; }
# 	</style>
# 	""",
# 	unsafe_allow_html=True,
# )

try:
	resources = load_resources()
except FileNotFoundError as exc:
	st.error(f"Missing file: {exc}")
	st.stop()

if resources.get("shap_explainer") is None and resources.get("shap_error"):
	st.warning(resources["shap_error"])

bug_text = st.text_area(
	"Bug report",
	placeholder="Paste a bug report here...",
	height=90,
)
button_spacer, button_col = st.columns([4, 1])
with button_col:
	run_button = st.button("Report bug", type="primary", use_container_width=True)

col_left, col_right = st.columns([1, 1])

if run_button:
	if not bug_text.strip():
		st.warning("Please enter a bug report.")
	else:
		with st.spinner("Running inference..."):
			results = run_inference(bug_text, resources)

		if len(results["tokens"]) < MIN_WORDS:
			st.warning("This report is short after preprocessing and may reduce accuracy.")

		with col_left:
			
			plot_horizontal_bars(
				labels=results["top_labels"].tolist(),
				values=results["top_scores"].tolist(),
				title="Top 5 developers",
			)

		with col_right:
			
			plot_donut(
				labels=results["shap_labels"],
				values=results["shap_scores"],
				title="Top 5 Features",
			)
			if results.get("shap_error"):
				st.warning(f"SHAP failed: {results['shap_error']}")
