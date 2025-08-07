class DataStore:
    def __init__(self):
        self.data = {}
    def save(self, name ,value):
        self.data[name] = value
    def get(self,name):
        return self.data.get(name,None)
    def delete(self,name):
        if name in self.data:
            del self.data[name]