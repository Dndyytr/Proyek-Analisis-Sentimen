import json
import re
import unicodedata
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CS2ReviewClassifier:

    def __init__(self, project_dir, device=None):

        self.project_dir = Path(project_dir)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        with open(
            self.project_dir / "reports/ensemble_bert_roberta/ensemble_best_config.json"
        ) as f:
            self.config = json.load(f)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            str(self.project_dir / "models/bert_tuned_best")
        )

        self.roberta_tokenizer = AutoTokenizer.from_pretrained(
            str(self.project_dir / "models/roberta_base_cs2")
        )

        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            str(self.project_dir / "models/bert_tuned_best")
        ).to(self.device).eval()

        self.roberta_model = AutoModelForSequenceClassification.from_pretrained(
            str(self.project_dir / "models/roberta_base_cs2")
        ).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, text):

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Review tidak boleh kosong.")

        model_text = unicodedata.normalize("NFKC", text)
        model_text = model_text.casefold()
        model_text = re.sub(r"\s+", " ", model_text).strip()

        bert_inputs = self.bert_tokenizer(
            model_text, return_tensors="pt",
            truncation=True, padding=True, max_length=192
        )

        roberta_inputs = self.roberta_tokenizer(
            model_text, return_tensors="pt",
            truncation=True, padding=True, max_length=192
        )

        bert_inputs = {
            k: v.to(self.device) for k, v in bert_inputs.items()
        }

        roberta_inputs = {
            k: v.to(self.device) for k, v in roberta_inputs.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.device.type == "cuda"
        ):
            bert_output = self.bert_model(**bert_inputs)
            roberta_output = self.roberta_model(**roberta_inputs)

        bert_prob = torch.softmax(
            bert_output.logits.float(), dim=1
        )[0, 1].item()

        roberta_prob = torch.softmax(
            roberta_output.logits.float(), dim=1
        )[0, 1].item()

        ensemble_prob = (
            self.config["bert_weight"] * bert_prob
            + self.config["roberta_weight"] * roberta_prob
        )

        threshold = self.config["threshold"]

        label = "positif" if ensemble_prob >= threshold else "negatif"

        return {
            "review": text,
            "predicted_label": label,
            "positive_probability": round(ensemble_prob, 4),
            "negative_probability": round(1 - ensemble_prob, 4),
            "bert_positive_probability": round(bert_prob, 4),
            "roberta_positive_probability": round(roberta_prob, 4),
            "threshold": threshold
        }
