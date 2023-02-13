import copy
from collections import namedtuple
import operator
import math


ProxyOperation = namedtuple('ProxyOperation', ['func', 'args', 'kwargs'], defaults=[(), {}])


def _operator_call(obj, /, *args, **kwargs): return obj(*args, **kwargs)
def _operator_radd(a, b): return b + a
def _operator_rsub(a, b): return b - a
def _operator_rmul(a, b): return b * a
def _operator_rmatmul(a, b): return b @ a
def _operator_rtruediv(a, b): return b / a
def _operator_rfloordiv(a, b): return b // a
def _operator_rmod(a, b): return b % a
def _operator_rdivmod(a, b): return divmod(b, a)
def _operator_rpow(a, b): return b ** a
def _operator_rlshift(a, b): return b << a
def _operator_rrshift(a, b): return b >> a
def _operator_rand(a, b): return b & a
def _operator_rxor(a, b): return b ^ a
def _operator_ror(a, b): return b | a


def _op_appender(func):
    def _wrapped(proxy, *args, **kwargs):
        return Proxy.append_op(proxy, ProxyOperation(func, args, kwargs))
    return _wrapped


def _format_op(text, op):
    fargs = [repr(arg) for arg in op.args]
    fargs += [f'{key}={value!r}' for key, value in op.kwargs.items()]
    fargs = ', '.join(fargs)
    mapping = {
        operator.lt: '{text} < {fargs}',
        operator.le: '{text} <= {fargs}',
        operator.eq: '{text} == {fargs}',
        operator.ne: '{text} != {fargs}',
        operator.gt: '{text} > {fargs}',
        operator.ge: '{text} >= {fargs}',

        getattr: '{text}.{args[0]!s}', #Use original arg as string
        operator.getitem: '{text}[{fargs}]',
        operator.contains: '{fargs} in {text}',
        _operator_call: '{text}({fargs})',

        operator.add: '{text} + {fargs}',
        operator.sub: '{text} - {fargs}',
        operator.mul: '{text} * {fargs}',
        operator.matmul: '{text} @ {fargs}',
        operator.truediv: '{text} / {fargs}',
        operator.floordiv: '{text} // {fargs}',
        operator.mod: '{text} % {fargs}',
        divmod: 'divmod({text}, {fargs})',
        operator.pow: '{text} ** {fargs}',
        operator.lshift: '{text} << {fargs}',
        operator.rshift: '{text} >> {fargs}',
        operator.and_: '{text} & {fargs}',
        operator.xor: '{text} ^ {fargs}',
        operator.or_: '{text} | {fargs}',

        _operator_radd: '{fargs} + {text}',
        _operator_rsub: '{fargs} - {text}',
        _operator_rmul: '{fargs} * {text}',
        _operator_rmatmul: '{fargs} @ {text}',
        _operator_rtruediv: '{fargs} / {text}',
        _operator_rfloordiv: '{fargs} // {text}',
        _operator_rmod: '{fargs} % {text}',
        _operator_rdivmod: 'divmod({fargs}, {text})',
        _operator_rpow: '{fargs} ** {text}',
        _operator_rlshift: '{fargs} << {text}',
        _operator_rrshift: '{fargs} >> {text}',
        _operator_rand: '{fargs} & {text}',
        _operator_rxor: '{fargs} ^ {text}',
        _operator_ror: '{fargs} | {text}',

        operator.neg: '-{text}',
        operator.pos: '+{text}',
        operator.abs: 'abs({text})',
        operator.invert: '~{text}',

        complex: 'complex({text})',
        int: 'int({text})',
        float: 'float({text})',

        operator.index: 'operator.index({text})',

        float: 'round({text})',
        math.trunc: 'math.trunc({text})',
        math.floor: 'math.floor({text})',
        math.ceil: 'math.ceil({text})',
    }
    template = mapping.get(op.func)
    if template is None:
        return f'{text}.{op.func!s}({fargs})'
    else:
        return template.format(text=text, fargs=fargs, args=op.args)


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
            op.func in [getattr, operator.getitem]
            for op in proxy.__operations__
        )

    def is_dummy(cls, proxy):
        if len(proxy.__operations__) != 1:
            return False

        op = proxy.__operations__[0]
        return op.func is getattr and op.args[0] == '_'

    def is_skip(cls, proxy):
        if len(proxy.__operations__) != 1:
            return False

        op = proxy.__operations__[0]
        return op.func in [getattr, operator.getitem] and op.args[0] == 'skip'

    def is_lambda(cls, proxy):
        if len(proxy.__operations__) != 1:
            return False

        op = proxy.__operations__[0]
        return op.func is operator.getitem and callable(op.args[0])

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
            if op.func is getattr:
                setattr(obj, op.args[0], value)
            elif op.func is operator.getitem:
                obj[op.args[0]] = value

            return orig_obj


class Proxy(metaclass=ProxyMeta):
    def __init__(self, *operations):
        self.__operations__ = operations

    def __hash__(self):
        hashable_funcs = [getattr, operator.getitem]
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

    __lt__ = _op_appender(operator.lt)
    __le__ = _op_appender(operator.le)
    __eq__ = _op_appender(operator.eq)
    __ne__ = _op_appender(operator.ne)
    __gt__ = _op_appender(operator.gt)
    __ge__ = _op_appender(operator.ge)

    __getattr__ = _op_appender(getattr)
    __call__ = _op_appender(_operator_call)
    __getitem__ = _op_appender(operator.getitem)
    __contains__ = _op_appender(operator.contains)

    __add__ = _op_appender(operator.add)
    __sub__ = _op_appender(operator.sub)
    __mul__ = _op_appender(operator.mul)
    __matmul__ = _op_appender(operator.matmul)
    __truediv__ = _op_appender(operator.truediv)
    __floordiv__ = _op_appender(operator.floordiv)
    __mod__ = _op_appender(operator.mod)
    __divmod__ = _op_appender(divmod)
    __pow__ = _op_appender(operator.pow)
    __lshift__ = _op_appender(operator.lshift)
    __rshift__ = _op_appender(operator.rshift)
    __and__ = _op_appender(operator.and_)
    __xor__ = _op_appender(operator.xor)
    __or__ = _op_appender(operator.or_)

    __radd__ = _op_appender(_operator_radd)
    __rsub__ = _op_appender(_operator_rsub)
    __rmul__ = _op_appender(_operator_rmul)
    __rmatmul__ = _op_appender(_operator_rmatmul)
    __rtruediv__ = _op_appender(_operator_rtruediv)
    __rfloordiv__ = _op_appender(_operator_rfloordiv)
    __rmod__ = _op_appender(_operator_rmod)
    __rdivmod__ = _op_appender(_operator_rdivmod)
    __rpow__ = _op_appender(_operator_rpow)
    __rlshift__ = _op_appender(_operator_rlshift)
    __rrshift__ = _op_appender(_operator_rrshift)
    __rand__ = _op_appender(_operator_rand)
    __rxor__ = _op_appender(_operator_rxor)
    __ror__ = _op_appender(_operator_ror)

    __iadd__ = _op_appender(operator.iadd)
    __isub__ = _op_appender(operator.isub)
    __imul__ = _op_appender(operator.imul)
    __imatmul__ = _op_appender(operator.imatmul)
    __itruediv__ = _op_appender(operator.itruediv)
    __ifloordiv__ = _op_appender(operator.ifloordiv)
    __imod__ = _op_appender(operator.imod)
    __ipow__ = _op_appender(operator.ipow)
    __ilshift__ = _op_appender(operator.ilshift)
    __irshift__ = _op_appender(operator.irshift)
    __iand__ = _op_appender(operator.iand)
    __ixor__ = _op_appender(operator.ixor)
    __ior__ = _op_appender(operator.ior)

    __neg__ = _op_appender(operator.neg)
    __pos__ = _op_appender(operator.pos)
    __abs__ = _op_appender(operator.abs)
    __invert__ = _op_appender(operator.invert)

    __complex__ = _op_appender(complex)
    __int__ = _op_appender(int)
    __float__ = _op_appender(float)

    __index__ = _op_appender(operator.index)

    __round__ = _op_appender(round)
    __trunc__ = _op_appender(math.trunc)
    __floor__ = _op_appender(math.floor)
    __ceil__ = _op_appender(math.ceil)


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
