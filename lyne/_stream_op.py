from ._proxy import *
from ._stream_item import *
import copy


class Operation:
    __slots__ = ('func', 'args', 'kwargs', 'output', 'skip')

    def __init__(self, func, *args, **kwargs):
        output = kwargs.pop('__output', None)
        skip = kwargs.pop('__skip', True)

        assert callable(func), "func must be callable"
        assert Proxy.single_type_in(args + tuple(kwargs.values())), "Proxies must all be of same type"

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.output = OutputMapping(output)
        if skip and not callable(skip):
            skip = lambda item: item.skip
        self.skip = skip

    def __repr__(self):
        args = [repr(arg) for arg in self.args]
        args += [f'{key}={value!r}' for key, value in self.kwargs.items()]
        return f'{self.func.__name__}({", ".join(args)}) >> {self.output}'

    def _get_ops(self):
        if hasattr(self, 'operations'):
            return list(self.operations)
        else:
            return [self]

    @property
    def ignore_skipped(self):
        return self(__skip=True)

    @property
    def only_skipped(self):
        return self(__skip=lambda item: not item.skip)

    @property
    def all(self):
        return self(__skip=False)

    def __call__(self, *args, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        return self.using(*self.args, *args, **kwargs)

    def using(self, *args, **kwargs):
        kwargs.setdefault('__output', self.output)
        kwargs.setdefault('__skip', self.skip)
        return self.__class__(self.func, *args, **kwargs)

    def process(self, *args, **kwargs):
        if not args:
            stream = Stream([Item()])
        else:
            stream, *args = args
            if not isinstance(stream, Stream):
                stream = Stream([stream])
        new_kwargs = {**self.kwargs, **kwargs}
        return Stream(self._call_gen(stream, *self.args, *args, **new_kwargs))

    def _call_gen(self, stream, *args, **kwargs):
        item_args = ItemProxy.instance_in((self.func,) + args + tuple(kwargs.values()))
        if item_args:
            for item in stream:
                if self.skip and self.skip(item):
                    yield item
                else:
                    new_args = Proxy.apply(item, args)
                    new_kwargs = Proxy.apply(item, kwargs)
                    result = Proxy.get_value(self.func, item) if isinstance(self.func, Proxy) else self.func
                    if callable(result):
                        result = result(*new_args, **new_kwargs)
                    yield from self.output.gen_result(item, result)
        else:
            args = Proxy.apply(stream, args)
            kwargs = Proxy.apply(stream, kwargs)
            func = Proxy.get_value(self.func, stream) if isinstance(self.func, Proxy) else self.func
            result = self.func(*args, **kwargs)
            yield from self.output.gen_result(stream, result)

    def __rshift__(self, other):
        return self(__output=other)

    def __or__(self, other):
        ops = self._get_ops()
        if isinstance(other, Operation):
            ops += other._get_ops()
        else:
            return NotImplemented

        return SequentialOperation(ops)


class SequentialOperation(Operation):
    def __init__(self, operations):
        self.operations = operations

    def __repr__(self):
        return f'{self.__class__.__name__}({self.operations!r})'

    def __call__(self, *args, **kwargs):
        first_op, *other_ops = self.operations
        first_op = first_op(*args, **kwargs)
        return self.__class__([first_op, *other_ops])

    def using(self, *args, **kwargs):
        first_op, *other_ops = self.operations
        first_op = first_op.using(*args, **kwargs)
        return self.__class__([first_op, *other_ops])

    def process(self, *args, **keywords):
        ops = iter(self.operations)
        op = next(ops)
        #Execute first operation with passed args and keywords.
        stream = op.process(*args, **keywords)
        #Then, execute all other operations with the preceding one as input.
        for op in ops:
            stream = op.process(stream)

        return stream


class OutputMapping(dict):
    def __init__(self, mapping):
        if mapping is None:
            mapping = {}
        elif isinstance(mapping, dict):
            pass
        elif isinstance(mapping, (list, tuple)):
            mapping = {
                proxy: OutputProxy()[index]
                for index, proxy in enumerate(mapping)
            }
        else:
            mapping = {mapping: OutputProxy()}

        assert all(isinstance(key, AssignableProxy) for key in mapping.keys()), "All mapping keys must be of type AssignableProxy"
        assert all(isinstance(value, OutputProxy) for value in mapping.values()), "All mapping values must be of type OutputProxy"
        super().__init__(mapping)

    def gen_result(self, obj, result):
        if ItemProxy.instance_in(self):
            new_item = obj if isinstance(obj, Item) else Item()
            yield self.apply(new_item, result)
        elif StreamProxy.instance_in(self):
            for value in result:
                new_item = copy.deepcopy(obj) if isinstance(obj, Item) else Item()
                yield self.apply(new_item, value)
        elif isinstance(obj, Stream):
            yield from obj
        else:
            yield obj

    def apply(self, obj, result):
        for key, value in self.items():
            if Proxy.is_dummy(key):
                continue
            assign_val = Proxy.get_value(value, result)
            if Proxy.is_skip(key) and not assign_val:
                continue
            obj = Proxy.set_value(key, obj, assign_val)
        return obj


class OperationDecorator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, func, *args, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        return Operation(func, *self.args, *args, **kwargs)

    def using(self, *args, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        return self.__class__(*self.args, *args, **kwargs)

    def __getitem__(self, func):
        return self(func)

    def __rshift__(self, other):
        kwargs = self.kwargs.copy()
        kwargs['__output'] = other
        return self.__class__(*self.args, **kwargs)
