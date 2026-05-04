import json
import random
from app.inference.emotion_predictor import EmotionPredictor
from app.inference.gemini_predictor import GeminiPredictor
from app.inference.safety_filter import SafetyFilter
from app.inference.conversation_memory import ConversationMemory
from app.inference.response_cleaner import ResponseCleaner



class PersonalizationEngine:
    def __init__(self):
        self.profile = {
            "name": None,
            "last_emotion": None
        }

    def update(self, user_input, emotion):
        self.profile["last_emotion"] = emotion

        if "my name is" in user_input.lower():
            self.profile["name"] = user_input.split("my name is")[-1].strip()

    def apply(self, response):
        if self.profile["name"]:
            return f"{self.profile['name']}, {response}"
        return response


class ResponseGenerator:

    def __init__(self):

        self.emotion_model = EmotionPredictor()
        self.dialog_model = GeminiPredictor()
        self.safety_filter = SafetyFilter()
        self.memory = ConversationMemory()
        self.cleaner = ResponseCleaner()

        # ✅ NEW: personalization
        self.personalization = PersonalizationEngine()

        with open("data/coping_strategies.json", "r", encoding="utf-8") as f:
            self.coping_strategies = json.load(f)

    def get_primary_emotion(self, emotions):
        return emotions[0]["emotion"] if emotions else "neutral"

    # --------------------------------
    # HUMAN FLOW CONTROL (FIXED)
    # --------------------------------
    def humanize(self, response, emotion):

        bad = ["User:", "Bot:", "Assistant:", "AI:"]
        for b in bad:
            response = response.replace(b, "")

        response = response.strip()

        sentences = response.split(". ")
        if len(sentences) > 4:
            response = ". ".join(sentences[:4])
            if not response.endswith("."):
                response += "."

        if response.count("?") > 1:
            response = response.split("?")[0] + "?"

        if emotion == "sadness":
            if "?" not in response:
                response += " Do you want to talk about it?"

        elif emotion == "anxiety":
            if len(response.split()) > 25:
                response = response[:150].rsplit(" ", 1)[0] + "..."

        return response.strip()

    # --------------------------------
    # MAIN GENERATE FUNCTION
    # --------------------------------
    def generate(self, user_input):

        
        if self.safety_filter.check_crisis(user_input):
            safe_msg = self.safety_filter.safe_response()
            return {
                "emotion": "crisis",
                "response": safe_msg
            }

        # 1. Emotion detection
        emotions = self.emotion_model.predict_emotions(user_input)
        primary_emotion = self.get_primary_emotion(emotions)
        primary_emotion = primary_emotion.lower().strip()

        
        if "fail" in user_input.lower():
            primary_emotion = "sadness"


        context = self.memory.get_context()

    
        response = self.dialog_model.generate_response(
            user_input=user_input,
            emotion=primary_emotion,
            context=context
        )

        # 4. Humanize
        response = self.humanize(response, primary_emotion)

        
        self.personalization.update(user_input, primary_emotion)

        
        strategies = self.coping_strategies.get(primary_emotion, [])

        if primary_emotion in ["sadness", "stress", "anxiety"] and strategies:
            if random.random() < 0.4:
                strategy = random.choice(strategies)
                response += f"\n\n💡 {strategy}"

        
        response = self.cleaner.clean(response)

        
        response = self.safety_filter.filter_response(user_input, response)

    
        response = self.personalization.apply(response)


        self.memory.add_turn(user_input, response)

        return {
            "emotion": primary_emotion,
            "response": response
        }