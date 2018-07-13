class A(object):

    def __init__(self, v):
        self.val = v

    def get_value(self):
        return self.val

a = A(42)
b = a.value