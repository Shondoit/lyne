import copy
from collections import namedtuple


ProxyOperation = namedtuple('ProxyOperation', ['func', 'args', 'kwargs'], defaults=[(), {}])


def _op_appender(func):
    def _wrapped(proxy, *args, **kwargs):
        return Proxy.append_op(proxy, ProxyOperation(func, args, kwargs))
    return _wrapped


def _format_op(text, op):
    fargs = [repr(arg) for arg in op.args]
    fargs += [f'{key}={value!r}' for key, value in op.kwargs.items()]
    fargs = ', '.join(fargs)
    mapping = {
        '__lt__': '{text} < {fargs}',
        '__le__': '{text} <= {fargs}',
        '__eq__': '{text} == {fargs}',
        '__ne__': '{text} != {fargs}',
        '__gt__': '{text} > {fargs}',
        '__ge__': '{text} >= {fargs}',

        getattr: '{text}.{op.args[0]!s}', #Use original arg as string
        '__getitem__': '{text}[{fargs}]',
        '__contains__': '{fargs} in {text}',
        '__call__': '{text}({fargs})',

        '__add__': '{text} + {fargs}',
        '__sub__': '{text} - {fargs}',
        '__mul__': '{text} * {fargs}',
        '__matmul__': '{text} @ {fargs}',
        '__truediv__': '{text} / {fargs}',
        '__floordiv__': '{text} // {fargs}',
        '__mod__': '{text} % {fargs}',
        '__divmod__': 'divmod({text}, {fargs})',
        '__pow__': '{text} ** {fargs}',
        '__lshift__': '{text} << {fargs}',
        '__rshift__': '{text} >> {fargs}',
        '__and__': '{text} & {fargs}',
        '__xor__': '{text} ^ {fargs}',
        '__or__': '{text} | {fargs}',

        '__radd__': '{fargs} + {text}',
        '__rsub__': '{fargs} - {text}',
        '__rmul__': '{fargs} * {text}',
        '__rmatmul__': '{fargs} @ {text}',
        '__rtruediv__': '{fargs} / {text}',
        '__rfloordiv__': '{fargs} // {text}',
        '__rmod__': '{fargs} % {text}',
        '__rdivmod__': 'divmod({fargs}, {text})',
        '__rpow__': '{fargs} ** {text}',
        '__rlshift__': '{fargs} << {text}',
        '__rrshift__': '{fargs} >> {text}',
        '__rand__': '{fargs} & {text}',
        '__rxor__': '{fargs} ^ {text}',
        '__ror__': '{fargs} | {text}',

        '__neg__': '-{text}',
        '__pos__': '+{text}',
        '__abs__': 'abs({text})',
        '__invert__': '~{text}',

        '__complex__': 'complex({text})',
        '__int__': 'int({text})',
        '__float__': 'float({text})',
        '__index__': 'operator.index({text})',

        '__float__': 'round({text})',
        '__trunc__': 'math.trunc({text})',
        '__floor__': 'math.floor({text})',
        '__ceil__': 'math.ceil({text})',
    }
    template = mapping.get(op.func)
    if template is None:
        return f'{text}.{op.func}({fargs})'
    else:
        return template.format(text=text, fargs=fargs, args=args, kwargs=kwargs)


class ProxyMeta(type):
    def __instancecheck__(cls, instance):
        type_ = type(instance)
        if not issubclass(type_, Proxy):
            return False
        elif issubclass(cls, AssignableProxy):
            return cls.is_assignable(instance) and not issubclass(type_, OutputProxy)
        elif issubclass(cls, InputProxy):
            return not issubclass(type_, OutputProxy)
        elif issubclass(cls, LambdaProxy):
            return cls.is_lambda(instance)
        else:
            return issubclass(type_, cls)


    def append_op(cls, proxy, op):
        return proxy.__class__(*proxy.__operations__, op)

    def instance_in(cls, iterable):
        return any(isinstance(obj, cls) for obj in iterable)    

    def single_type_in(cls, iterable):
        return len(set(type(obj) for obj in iterable if isinstance(obj, cls))) <= 1

    def is_assignable(cls, proxy):
        return all(
            op.func in [getattr, '__getattribute__', '__getitem__']
            for op in proxy.__operations__
        )

    def is_dummy(cls, proxy):
        if len(proxy.__operations__) != 1:
            return False

        op = proxy.__operations__[0]
        return op.func in ['__getattribute__', getattr] and op.args[0] == '_'

    def is_skip(cls, proxy):
        if len(proxy.__operations__) != 1:
            return False

        op = proxy.__operations__[0]
        return op.func in [getattr, '__getattribute__', '__getitem__'] and op.args[0] == 'skip'

    def is_lambda(cls, proxy):
        if len(proxy.__operations__) != 1:
            return False

        op = proxy.__operations__[0]
        return op.func == '__getitem__' and callable(op.args[0])

    def apply(cls, obj, args):
        if isinstance(args, dict):
            return {
                key: Proxy.get_value(value, obj) if isinstance(value, Proxy) else value
                for key, value in args.items()
            }
        elif isinstance(args, (list, tuple)):
            return args.__class__(
                Proxy.get_value(arg, obj) if isinstance(arg, Proxy) else arg
                for arg in args
            )
        else:
            return Proxy.get_value(args, obj) if isinstance(args, Proxy) else args

    def get_value(cls, proxy, obj):
        if not isinstance(proxy, Proxy):
            return proxy
        elif isinstance(proxy, LambdaProxy):
            op = proxy.__operations__[0]
            return op.args[0](obj)
        else:
            for op in proxy.__operations__:
                if isinstance(op.func, str):
                    func = getattr(obj, op.func)
                    obj = func(*op.args, **op.kwargs)
                else:
                    obj = op.func(obj, *op.args, **op.kwargs)
            return obj

    def set_value(cls, proxy, obj, value):
        assert isinstance(proxy, AssignableProxy), "Cannot assign to this Proxy instance"
        if not proxy.__operations__:
            return value
        else:
            orig_obj = obj
            for op in proxy.__operations__[:-1]:
                if isinstance(op.func, str):
                    func = getattr(obj, op.func)
                    obj = func(*op.args, **op.kwargs)
                else:
                    obj = op.func(obj, *op.args, **op.kwargs)

            op = proxy.__operations__[-1]
            if op.func in [getattr, '__getattribute__']:
                setattr(obj, op.args[0], value)
            elif op.func == '__getitem__':
                obj[op.args[0]] = value

            return orig_obj


class Proxy(metaclass=ProxyMeta):
    def __init__(self, *operations):
        self.__operations__ = operations

    def __hash__(self):
        hashable_funcs = [getattr, '__getattribute__', '__getitem__']
        if not isinstance(self, AssignableProxy):
            unhashable_ops = ', '.join(
                op.func for op in self.__operations__
                if op.func not in hashable_funcs
            )
            return TypeError(f'unhashable operations in instance: {unhashable_ops}')
        else:
            return hash(tuple(
                (str(op.func), op.args)
                for op in self.__operations__
                if op.func in hashable_funcs
            ))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__operations__})'

    def __str__(self):
        #result = '<...>'
        result = self.__class__.__name__[0]
        for op in self.__operations__:
            result = _format_op(result, op)
        return result

    __lt__ = _op_appender('__lt__')
    __le__ = _op_appender('__le__')
    __eq__ = _op_appender('__eq__')
    __ne__ = _op_appender('__ne__')
    __gt__ = _op_appender('__gt__')
    __ge__ = _op_appender('__ge__')

    #Call main entry function instead of __getattr__
    __getattr__ = _op_appender(getattr)
    __call__ = _op_appender('__call__')
    __getitem__ = _op_appender('__getitem__')
    __contains__ = _op_appender('__contains__')

    __add__ = _op_appender('__add__')
    __sub__ = _op_appender('__sub__')
    __mul__ = _op_appender('__mul__')
    __matmul__ = _op_appender('__matmul__')
    __truediv__ = _op_appender('__truediv__')
    __floordiv__ = _op_appender('__floordiv__')
    __mod__ = _op_appender('__mod__')
    __divmod__ = _op_appender('__divmod__')
    __pow__ = _op_appender('__pow__')
    __lshift__ = _op_appender('__lshift__')
    __rshift__ = _op_appender('__rshift__')
    __and__ = _op_appender('__and__')
    __xor__ = _op_appender('__xor__')
    __or__ = _op_appender('__or__')

    __radd__ = _op_appender('__radd__')
    __rsub__ = _op_appender('__rsub__')
    __rmul__ = _op_appender('__rmul__')
    __rmatmul__ = _op_appender('__rmatmul__')
    __rtruediv__ = _op_appender('__rtruediv__')
    __rfloordiv__ = _op_appender('__rfloordiv__')
    __rmod__ = _op_appender('__rmod__')
    __rdivmod__ = _op_appender('__rdivmod__')
    __rpow__ = _op_appender('__rpow__')
    __rlshift__ = _op_appender('__rlshift__')
    __rrshift__ = _op_appender('__rrshift__')
    __rand__ = _op_appender('__rand__')
    __rxor__ = _op_appender('__rxor__')
    __ror__ = _op_appender('__ror__')

    #Replace in-place operator with normal operator
    __iadd__ = _op_appender('__add__')
    __isub__ = _op_appender('__sub__')
    __imul__ = _op_appender('__mul__')
    __imatmul__ = _op_appender('__matmul__')
    __itruediv__ = _op_appender('__truediv__')
    __ifloordiv__ = _op_appender('__floordiv__')
    __imod__ = _op_appender('__mod__')
    __ipow__ = _op_appender('__pow__')
    __ilshift__ = _op_appender('__lshift__')
    __irshift__ = _op_appender('__rshift__')
    __iand__ = _op_appender('__and__')
    __ixor__ = _op_appender('__xor__')
    __ior__ = _op_appender('__or__')

    __neg__ = _op_appender('__neg__')
    __pos__ = _op_appender('__pos__')
    __abs__ = _op_appender('__abs__')
    __invert__ = _op_appender('__invert__')

    __complex__ = _op_appender('__complex__')
    __int__ = _op_appender('__int__')
    __float__ = _op_appender('__float__')

    __index__ = _op_appender('__index__')

    __round__ = _op_appender('__round__')
    __trunc__ = _op_appender('__trunc__')
    __floor__ = _op_appender('__floor__')
    __ceil__ = _op_appender('__ceil__')


class AbstractProxy(Proxy):
    def __new__(cls, *args, **kwargs):
        raise TypeError(f"'{cls.__name__}' is an abstract class and may not be instantiated")


class AssignableProxy(AbstractProxy):
    pass


class InputProxy(AbstractProxy):
    pass


class LambdaProxy(AbstractProxy):
    pass


class StreamProxy(Proxy):
    pass


class ItemProxy(Proxy):
    pass


class OutputProxy(Proxy):
    pass
