class Patient:
    def __init__(self, sex: str, age: int, bmi: float):
        self.sex = sex
        self.age = age
        self.bmi = bmi
        self.hormones = self._initialize_hormones()

    def _initialize_hormones(self) -> dict:
        if self.sex == "female":
            return {"estrogen": 100 * (1 + 0.1 * (self.bmi - 25)), "testosterone": 20}
        else:
            return {"estrogen": 30, "testosterone": 150 * (1 + 0.05 * (self.bmi - 25))}