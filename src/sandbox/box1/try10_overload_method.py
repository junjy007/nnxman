class A:
    def f(self):
        print "A:f"
class B:
    def f(self, n):
        print "B:f:{}".format(n)

a=A()
b=B()
a.f()
b.f(5)
b.f()
