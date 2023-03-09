class Station:
    def __init__(self, x, y, z=0.00):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"X= {self.x}\tY= {self.y}\tZ= {self.z}"

    def __repr__(self):
        return f"{str(self)}\n"