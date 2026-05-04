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