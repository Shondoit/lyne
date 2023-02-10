from ._stream_op import *
from ._stream_item import *
from ._proxy import *
from ._dims import *


S = StreamProxy()
I = ItemProxy()
O = OutputProxy()


Op = OperationDecorator()
Cond = Operation(lambda cond, reason: reason if cond else None) >> I.skip

Rel = RelativeValue(1.)
