import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentPredictor:

    def __init__(self, model_path=None):

        # Resolve model path: prefer HF_REPO_ID env var, fall back to local path
        if model_path is None:
            model_path = os.environ.get(
                "HF_REPO_ID", "models/intent_classifier_v4"
            )

        self.model_path = model_path

        # Read optional Hugging Face token for private repos
        hf_token = os.environ.get("HF_TOKEN")

        # Only pass token kwarg if a token is actually available
        token_kwargs = {"token": hf_token} if hf_token else {}

        # Ensure model directory exists (only relevant for local paths)
        os.makedirs("models", exist_ok=True)

        # Load device (GPU if available)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Intent Predictor running on:", self.device)
        print("Loading intent model from:", self.model_path)

        # Load tokenizer and model
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

    def predict_intent(self, text):

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

        predicted_index = logits.argmax(dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0][predicted_index].item()

        return {
            "intent": self.id2label[predicted_index],
            "confidence": float(confidence)
        }


if __name__ == "__main__":

    predictor = IntentPredictor()

    while True:
        text = input("\nEnter a message (or 'quit'): ")

        if text.lower() == "quit":
            break

        result = predictor.predict_intent(text)

        print(f"\nDetected intent: {result['intent']} ({result['confidence']:.2f})")
