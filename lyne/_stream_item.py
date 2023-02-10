class Stream:
    def __init__(self, iterable):
        self._iterable = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterable)


class Item(dict):
    __default__ = None

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.__default__

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __delitem__(self, key):
        try:
            super().__delitem__(key)
        except KeyError:
            pass

    def __getattr__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(name)
            else:
                return self.__default__

    def __setattr__(self, name, value):
        try:
            super().__getattribute__(name)
        except AttributeError:
            self.__setitem__(name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        try:
            super().__getattribute__(name)
        except AttributeError:
            self.__delitem__(name)
        else:
            super().__delattr__(name)
