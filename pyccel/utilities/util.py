from sympy.utilities.iterables import iterable


def do_once(method):
    "A decorator that runs a method only once."

    attrname = "_%s_result" % id(method)

    def decorated(self, *args, **kwargs):
        try:
            return getattr(self, attrname)
        except AttributeError:
            setattr(self, attrname, method(self, *args, **kwargs))
            return getattr(self, attrname)
    return decorated


iterate = lambda x: iter(x) if iterable(x) else iter([x])
