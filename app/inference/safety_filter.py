import re

class SafetyFilter:

    def __init__(self):
        self.crisis_keywords = [
            "suicide", "kill myself", "end my life", "die",
            "self harm", "hurt myself"
        ]

    def check_crisis(self, text):
        text = text.lower()
        return any(keyword in text for keyword in self.crisis_keywords)

    def safe_response(self):
        return (
            "I'm really sorry you're feeling this way. "
            "You are not alone. It might help to talk to someone you trust. "
            "If you can, please consider reaching out to a mental health professional or a support service."
        )

    def filter_response(self, user_input, generated_response):

        # Crisis detection (kept)
        if self.check_crisis(user_input):
            return self.safe_response()

        #
        generated_response = re.sub(
            r"\b(harm|kill|die)\b", "",
            generated_response,
            flags=re.IGNORECASE
        )

        return generated_response