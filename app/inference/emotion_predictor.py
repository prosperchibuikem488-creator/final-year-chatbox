import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmotionPredictor:

    def __init__(self, model_path=None, threshold=0.30):

        # Resolve model path: prefer HF_REPO_ID env var, fall back to local path
        if model_path is None:
            model_path = os.environ.get(
                "HF_REPO_ID", "models/emotion_classifier_v6"
            )

        self.model_path = model_path
        self.threshold = threshold

        # Read optional Hugging Face token for private repos
        hf_token = os.environ.get("HF_TOKEN")

        # Ensure model directory exists (only relevant for local paths)
        os.makedirs("models", exist_ok=True)

        # Load device (GPU if available)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Emotion Predictor running on:", self.device)
        print("Loading emotion model from:", self.model_path)

        # Load tokenizer and model — only forward the token kwarg when a
        # value is actually present so the transformers library can still
        # fall back to its own credential resolution when HF_TOKEN is unset.
        token_kwargs = {"token": hf_token} if hf_token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, **token_kwargs
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, **token_kwargs
        )

        self.model.to(self.device)
        self.model.eval()

        # Load label mapping
        self.id2label = self.model.config.id2label


    def predict_emotions(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits

        probs = torch.sigmoid(logits)
        probs = probs.squeeze().cpu().numpy()

        predicted_emotions = []

        for i, prob in enumerate(probs):
            if prob >= self.threshold:
                predicted_emotions.append({
                    "emotion": self.id2label[i],
                    "confidence": float(prob)
                })

        # fallback if nothing passes threshold
        if not predicted_emotions:
            max_index = probs.argmax()
            predicted_emotions.append({
                "emotion": self.id2label[max_index],
                "confidence": float(probs[max_index])
            })

        return predicted_emotions


if __name__ == "__main__":

    predictor = EmotionPredictor()

    while True:
        text = input("\nEnter a message (or 'quit'): ")

        if text.lower() == "quit":
            break

        emotions = predictor.predict_emotions(text)

        print("\nDetected emotions:")
        for e in emotions:
            print(f"{e['emotion']} ({e['confidence']:.2f})")