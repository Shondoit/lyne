from typing import Iterable


class Value:
    def __init__(self, value, sign=None):
        self.value = value
        self.sign = sign

    def __repr__(self):
        return repr(self.value)

    @classmethod
    def to_rel(cls, value, min_val, max_val=None):
        if isinstance(value, Iterable):
            if not isinstance(min_val, Iterable):
                min_val = [min_val] * len(value)
            if max_val is None:
                max_val = [None] * len(min_val)
            elif not isinstance(max_val, Iterable):
                max_val = [max_val] * len(value)

            return value.__class__(
                Value.to_rel(item, mn, mx)
                for item, mn, mx in zip(value, min_val, max_val)
            )
        elif isinstance(value, RelativeValue):
            return value
        else:
            if max_val is None:
                min_val, max_val = 0, min_val
            if value.sign == '+':
                max_val = max(0, max_val)
                min_val = 0
            elif value.sign == '-':
                min_val = min(0, min_val)
                max_val = 0
            return RelativeValue((value - min_val) / (max_val - min_val))

    @classmethod
    def to_abs(cls, value, min_val, max_val=None, dtype=None):
        if isinstance(value, Iterable):
            if not isinstance(min_val, Iterable):
                min_val = [min_val] * len(value)
            if max_val is None:
                max_val = [None] * len(min_val)
            elif not isinstance(max_val, Iterable):
                max_val = [max_val] * len(value)

            return value.__class__(
                Value.to_abs(item, mn, mx, dtype)
                for item, mn, mx in zip(value, min_val, max_val)
            )
        else:
            if isinstance(value, RelativeValue):
                if max_val is None:
                    min_val, max_val = 0, min_val
                if value.sign == '+':
                    max_val = max(0, max_val)
                    min_val = 0
                elif value.sign == '-':
                    min_val = min(0, min_val)
                    max_val = 0

                result = min_val + value.value * (max_val - min_val)
            else:
                result = value

            return dtype(result) if dtype else result


class RelativeValue(Value):
    @property
    def pos(self):
        return self.__class__(self.value, '+')

    @property
    def neg(self):
        return self.__class__(self.value, '-')

    def __rmod__(self, value):
        return self.__class__(value / 100., self.sign)

    def __call__(self, value, sign=...):
        return self.__class__(value, self.sign if sign == ... else sign)

    def __repr__(self):
        val = format(self.value * 100., '.10f')
        val = val.rstrip('0')
        val = val.rstrip('.')
        sign = self.sign or ''
        return f'{sign}{val}%Rel'
        #return f'{self.__class__.__name__}({self.value})'
