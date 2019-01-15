import types
import dis
import builtins
import sys
import operator
import collections


class Frame(object):

    def __init__(self, code, locals_nm, globals_nm, prev_frame):
        self.code = code
        self.globals = globals_nm
        self.locals = locals_nm
        self.prev_frame = prev_frame
        self.stack = []
        self.block_stack = []
        self.builtins = prev_frame.builtins if prev_frame else builtins.__dict__
        self.cur_instruction_idx = 0
        self.co_names = code.co_names
        self.co_consts = code.co_consts
        self.co_varnames = code.co_varnames
        self.instructions = list(dis.get_instructions(code))


class Block(object):

    def __init__(self, type, delta, frame):
        self.type = type
        self.delta = delta
        self.frame = frame


def equal_type(l, r):
    return isinstance(l, type(r))


class VirtualMachine:

    def __init__(self, logging=False):
        self.logging = logging
        self.frame = None
        self.frames = []
        self.exception = None
        self.value = None

        self.operations = {
            'POWER': operator.pow,
            'MULTIPLY': operator.mul,
            'MATRIX_MULTIPLY': operator.matmul,
            'FLOOR_DIVIDE': operator.floordiv,
            'TRUE_DIVIDE': operator.truediv,
            'MODULO': operator.mod,
            'ADD': operator.add,
            'SUBTRACT': operator.sub,
            'SUBSCR': operator.getitem,
            'LSHIFT': operator.lshift,
            'RSHIFT': operator.rshift,
            'AND': operator.and_,
            'XOR': operator.xor,
            'OR': operator.or_,
        }

        self.compare_funcs = {
            '<': operator.lt,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            '>=': operator.ge,
            '>': operator.gt,
            'in': lambda x, y: x in y,
            'not in': lambda x, y: x not in y,
            'is': lambda x, y: x is y,
            'is not': lambda x, y: x is not y,
            'exception match': equal_type,
        }

    def log(self, *to_prints, in_line=False):
        if self.logging:
            if in_line:
                print(*to_prints)
            else:
                for to_print in to_prints:
                    print(to_print)

    def run(self, code: types.CodeType) -> None:
        """
        :param code: code for interpreting
        """
        self.log(*list(dis.get_instructions(code)))
        frame = Frame(code, {}, {}, None)
        self.run_frame(frame)

    def pop_frame(self):
        self.frames.pop()
        self.frame = self.frames[-1] if self.frames else None

    def push_frame(self, frame):
        self.frames.append(frame)
        self.frame = frame

    def push(self, *args):
        for arg in args:
            self.frame.stack.append(arg)

    def pop(self):
        return self.frame.stack.pop()

    def popn(self, n):
        return list(reversed([self.pop() for _ in range(n)])) if n else []

    def top(self):
        return self.frame.stack[-1]

    def pop_block(self):
        return self.frame.block_stack.pop()

    def push_block(self, type, delta = None):
        self.frame.block_stack.append(Block(type, delta, self.frame))

    def get_byte_name_and_arg(self):
        instruction = self.frame.instructions[self.frame.cur_instruction_idx]
        byte_name = instruction.opname
        argument = instruction.arg
        self.frame.cur_instruction_idx += 1
        self.log(byte_name, argument, instruction.argval, in_line = True)
        return byte_name, argument

    def run_byte_func(self, byte_name, argument):
        try:
            result = None
            if byte_name.split("_")[0] in ["BINARY", "INPLACE"]:
                self.bin_inp_funcs(byte_name)
            else:
                bytecode_fn = getattr(self, byte_name)
                result = bytecode_fn() if argument is None else bytecode_fn(argument)
        except:
            result = 'exception'
            self.log(sys.exc_info())
            self.exception = sys.exc_info()[:2] + (None,)
        return result

    def handle_exception(self):
        if self.frame.block_stack:
            block = self.pop_block()
            if block.type == "except":
                self.push_block('except-handler')
                exctype, value, tb = self.exception
                self.push(tb, value, exctype)
                self.push(tb, value, exctype)
                self.frame.cur_instruction_idx = block.delta
                return True
        return False

    def run_frame(self, frame):
        self.push_frame(frame)
        self.log("frame started")
        func_result = None
        while len(self.frame.instructions) > frame.cur_instruction_idx:
            byte_name, arg = self.get_byte_name_and_arg()
            self.log("stack before", len(self.frames), self.frame.stack)
            func_result = self.run_byte_func(byte_name, arg)
            if func_result == "exception":
                handled = self.handle_exception()
                if not handled:
                    exception, value, traceback = self.exception
                    exc = exception(value)
                    exc.__traceback__ = traceback
                    raise exc
            if func_result == "yield":
                func_result = False
                break

            self.log("stack after", self.frame.stack, "----------")
        self.log("frame ended")

        self.pop_frame()
        return func_result

    def find_associated(self, name, where_start="locals"):
        if name in self.frame.locals and where_start == "locals":
            val = self.frame.locals[name]
        elif name in self.frame.co_varnames:
            raise UnboundLocalError("local variable '%s' referenced before assignment" % name)
        elif name in self.frame.globals and (where_start == "locals" or where_start == "globals"):
            val = self.frame.globals[name]
        elif name in self.frame.builtins:
            val = self.frame.builtins[name]
        else:
            raise NameError("name '%s' is not defined" % name)
        self.log("find_associated:  ", name, val)
        return val

    def bin_inp_funcs(self, func_name):
        TOS1, TOS = self.popn(2)
        name = func_name.replace("BINARY_", "").replace("INPLACE_", "")
        self.push(self.operations[name](TOS1, TOS))

    def NOP(self):
        pass

    def POP_TOP(self):
        self.pop()

    def ROT_TWO(self):
        """Swaps the two top - most stack items."""
        tos1, tos = self.popn(2)
        self.push(tos)
        self.push(tos1)

    def ROT_THREE(self):
        """
        Lifts second and third stack item one position up, moves top down to position three.
        """
        tos2, tos1, tos = self.popn(3)
        self.push(tos, tos2, tos1)
        self.push(tos, tos2, tos1)

    def DUP_TOP(self):
        """Duplicates the reference on top of the stack."""
        self.push(self.top())

    def DUP_TOP_TWO(self):
        """Duplicates the two references on top of the stack,
        leaving them in the same order."""
        tos1, tos = self.popn(2)
        self.push(tos1, tos, tos1, tos)

    def UNARY_POSITIVE(self):
        """Implements TOS = +TOS."""
        self.push(+self.pop())

    def UNARY_NEGATIVE(self):
        """Implements TOS = -TOS."""
        self.push(-self.pop())

    def UNARY_NOT(self):
        """Implements TOS = not TOS."""
        self.push(not self.pop())

    def UNARY_INVERT(self):
        """Implements TOS = ~TOS."""
        self.push(~self.pop())

    def GET_ITER(self):
        """Implements TOS = iter(TOS)."""
        self.push(iter(self.pop()))

    def GET_YIELD_FROM_ITER(self):
        """
        If TOS is a generator iterator or coroutine object it is left as is.
        Otherwise, implements TOS = iter(TOS).
        """
        if not isinstance(self.top(), collections.Iterable):
            self.GET_ITER()

    def LOAD_NAME(self, namei):
        """Pushes the value associated with co_names[namei] onto the stack."""
        name = self.frame.co_names[namei]
        self.push(self.find_associated(name))

    def LOAD_CONST(self, consti):
        """Pushes co_consts[consti] onto the stack."""
        self.push(self.frame.co_consts[consti])

    def CALL_FUNCTION(self, argc):
        args = self.popn(argc)
        func = self.pop()
        val = func(*args)
        self.push(val)

    def RETURN_VALUE(self):
        return self.pop()

    def YIELD_VALUE(self):
        """Pops TOS and yields it from a generator."""
        self.value = self.pop()
        return "yield"

    def YIELD_FROM(self):
        """Pops TOS and delegates to it as a subiterator from a generator."""
        TOS = self.pop()
        raise NotImplementedError

    def BUILD_TUPLE(self, count):
        values = self.popn(count)
        self.push(tuple(values))

    def BUILD_LIST(self, count):
        self.push(self.popn(count))

    def BUILD_SET(self, count):
        values = self.popn(count)
        self.push(set(values))

    def BUILD_MAP(self, count):
        values = self.popn(count*2)
        self.push({values[2*i]: values[2*i+1] for i in range(count)})

    def BUILD_CONST_KEY_MAP(self, count):
        keys = self.pop()
        values = self.popn(count)
        self.push({key: value for key, value in zip(keys, values)})

    def BUILD_STRING(self, count):
        """Concatenates count strings from the stack and pushes the resulting string onto the stack."""
        self.push("".join(self.popn(count)))

    def BUILD_TUPLE_UNPACK(self, count):
        """
        Pops count iterables from the stack, joins them in a single tuple, and pushes the result.
        Implements iterable unpacking in tuple displays (*x, *y, *z).
        """
        result = []
        for _ in range(count):
            result += self.pop()
        self.push(tuple(result))

    def BUILD_TUPLE_UNPACK_WITH_CALL(self, count):
        """This is similar to BUILD_TUPLE_UNPACK, but is used for f(*x, *y, *z) call syntax.
        The stack item at position count + 1 should be the corresponding callable f.
        """
        result = []
        for _ in range(count):
            result += self.pop()
        self.push(self.pop()(tuple(result)))

    def BUILD_LIST_UNPACK(self, count):
        """This is similar to BUILD_TUPLE_UNPACK, but pushes a list instead of tuple.
        Implements iterable unpacking in list displays [*x, *y, *z].
        """
        result = []
        for _ in range(count):
            result += self.pop()
        self.push(result)

    def BUILD_SET_UNPACK(self, count):
        """This is similar to BUILD_TUPLE_UNPACK, but pushes a set instead of tuple.
        Implements iterable unpacking in set displays {*x, *y, *z}.
        """
        result = []
        for _ in range(count):
            result += self.pop()
        self.push(set(result))

    def BUILD_MAP_UNPACK(self, count):
        """Pops count mappings from the stack, merges them into a single dictionary, and pushes the result.
        Implements dictionary unpacking in dictionary displays {**x, **y, **z}.
        """
        result = {}
        for _ in range(count):
            result.update(self.pop())
        self.push(result)

    def BUILD_MAP_UNPACK_WITH_CALL(self, count):
        """This is similar to BUILD_MAP_UNPACK, but is used for f(**x, **y, **z) call syntax.
        The stack item at position count + 2 should be the corresponding callable f.
        The position of the callable is determined by adding 2 to the opcode argument
        instead of encoding it in the second byte of the argument.
        """
        result = {}
        for _ in range(count):
            result.update(self.pop())

        self.push(result)
        raise NotImplementedError

    def STORE_NAME(self, namei):
        """
        Implements name = TOS. namei is the index of name in the attribute co_names of the code object.
        The compiler tries to use STORE_FAST or STORE_GLOBAL if possible.
        """
        self.frame.locals[self.frame.co_names[namei]] = self.pop()

    def DELETE_NAME(self, namei):
        """Implements del name, where namei is the index into co_names attribute of the code object."""
        del self.frame.locals[self.frame.co_names[namei]]

    def UNPACK_SEQUENCE(self, count):
        seq = self.pop()
        self.push(*reversed(seq))

    def STORE_ATTR(self, namei):
        """Implements TOS.name = TOS1, where namei is the index of name in co_names."""
        TOS1, TOS = self.popn(2)
        setattr(TOS, self.frame.co_names[namei], TOS1)
        self.push(TOS)

    def DELETE_ATTR(self, namei):
        """Implements del TOS.name, using namei as index into co_names. """
        delattr(self.top(), self.frame.co_names[namei])

    def STORE_GLOBAL(self, namei):
        """Works as STORE_NAME, but stores the name as a global. """
        self.frame.globals[self.frame.co_names[namei]] = self.pop()

    def DELETE_GLOBAL(self, namei):
        """Works as DELETE_NAME, but deletes a global name."""
        del self.frame.globals[self.frame.co_names[namei]]

    def STORE_SUBSCR(self):
        """Implements TOS1[TOS] = TOS2. """
        TOS = self.pop()
        TOS1 = self.pop()
        TOS2 = self.pop()
        TOS1[TOS] = TOS2

    def DELETE_SUBSCR(self):
        """Implements del TOS1[TOS]. """
        TOS = self.pop()
        TOS1 = self.pop()
        del TOS1[TOS]

    def LOAD_ATTR(self, namei):
        """Replaces TOS with getattr(TOS, co_names[namei])."""
        TOS = self.pop()
        self.push(getattr(TOS, self.frame.co_names[namei]))

    def COMPARE_OP(self, opname):
        """Performs a Boolean operation. The operation name can be found in cmp_op[opname]."""
        l, r = self.popn(2)
        self.push(self.compare_funcs[dis.cmp_op[opname]](l, r))

    def LOAD_BUILD_CLASS(self):
        self.push(builtins.__build_class__)

    def PRINT_EXPR(self):
        print(self.pop())

    def LOAD_FAST(self, var_num):
        """Pushes a reference to the local co_varnames[var_num]
        onto the stack."""
        self.log(self.frame.co_varnames[var_num])
        self.log(self.frame.locals)
        self.log(self.frame.globals)
        self.push(self.find_associated(self.frame.co_varnames[var_num]))

    def STORE_FAST(self, var_num):
        """Stores TOS into the local co_varnames[var_num]."""
        self.frame.locals[self.frame.co_varnames[var_num]] = self.pop()

    def DELETE_FAST(self, var_num):
        """Deletes local co_varnames[var_num]."""
        del self.frame.locals[self.frame.co_varnames[var_num]]

    def STORE_ANNOTATION(self, namei):
        self.frame.locals['__annotations__'][self.frame.co_names[namei]] = self.pop()

    def CALL_FUNCTION_KW(self, argc):
        """Calls a callable object with positional (if any) and keyword arguments.
        argc indicates the total number of positional and keyword arguments.
        The top element on the stack contains a tuple of keyword argument names.
        Below that are keyword arguments in the order corresponding to the tuple.
        Below that are positional arguments, with the right-most parameter on top.
        Below the arguments is a callable object to call. CALL_FUNCTION_KW pops all
        arguments and the callable object off the stack, calls the callable object
        with those arguments, and pushes the return value returned by the callable object.
        Keyword arguments are packed in a tuple instead of a dictionary,
        argc indicates the total number of arguments.

        """
        keyword_arg_names = self.pop()
        values = self.popn(len(keyword_arg_names))
        kwargs = {key: value for key, value in zip(keyword_arg_names, values)}
        self.log(kwargs)
        args = self.popn(argc - len(keyword_arg_names))
        func = self.pop()
        self.push(func(*args, **kwargs))

    def CALL_FUNCTION_EX(self, flags):
        """Calls a callable object with variable set of positional and keyword arguments.
         If the lowest bit of flags is set, the top of the stack contains a mapping object
         containing additional keyword arguments.
          Below that is an iterable object containing positional arguments and a callable object to call.
           BUILD_MAP_UNPACK_WITH_CALL and BUILD_TUPLE_UNPACK_WITH_CALL can be used for
           merging multiple mapping objects and iterables containing arguments.
            Before the callable is called, the mapping object and iterable object are each “unpacked”
            and their contents passed in as keyword and positional arguments respectively.
            CALL_FUNCTION_EX pops all arguments and the callable object off the stack,
            calls the callable object with those arguments, and pushes the return value
            returned by the callable object.
        """
        kwargs = {}
        if flags & 0x01:
            kwargs = self.pop()
        func, args = self.popn(2)
        self.push(func(*args, **kwargs))

    def MAKE_FUNCTION(self, argc):
        """
        Pushes a new function object on the stack. From bottom to top,
        the consumed stack must consist of values if the argument carries a specified flag value
        - 0x01 a tuple of default values for positional-only and positional-or-keyword parameters in positional order
        - 0x02 a dictionary of keyword-only parameters’ default values
        - 0x04 an annotation dictionary
        - 0x08 a tuple containing cells for free variables, making a closure
        - the code associated with the function (at TOS1)
        - the qualified name of the function (at TOS)
        """
        code, name = self.popn(2)
        kwargs = {}
        if argc & 0x08:
            kwargs['func_closure'] = self.pop()
        if argc & 0x04:
            kwargs['func_annotations'] = self.pop()
        if argc & 0x02:
            kwargs['func_kwdefaults'] = self.pop()
        if argc & 0x01:
            kwargs['func_defaults'] = self.pop()

        func = Function(name, code, self.frame.globals, self, argc, kwargs)

        self.push(func)

    def BUILD_SLICE(self, argc):
        """
        Pushes a slice object on the stack. argc must be 2 or 3.
        If it is 2, slice(TOS1, TOS) is pushed; if it is 3, slice(TOS2, TOS1, TOS) is pushed.
        See the slice() built-in function for more information.
        """
        TOS1, TOS = self.popn(2)
        if argc == 2:
            self.push(slice(TOS1, TOS))
        elif argc == 3:
            self.push(slice(self.pop(), TOS1, TOS))

    def JUMP_FORWARD(self, delta):
        """Increments bytecode counter by delta."""
        self.frame.cur_instruction_idx += delta // 2
        self.log(self.frame.cur_instruction_idx)
        self.log("instractions", *self.frame.instructions)

    def POP_JUMP_IF_TRUE(self, target):
        """If TOS is true, sets the bytecode counter to target. TOS is popped."""
        if self.pop():
            self.frame.cur_instruction_idx = target // 2

    def POP_JUMP_IF_FALSE(self, target):
        """If TOS is false, sets the bytecode counter to target.TOS is popped."""
        if not self.pop():
            self.frame.cur_instruction_idx = target // 2

    def JUMP_IF_TRUE_OR_POP(self, target):
        """If TOS is true, sets the bytecode counter to target and leaves TOS on the stack.
        Otherwise(TOS is false), TOS is popped."""
        if self.top():
            self.frame.cur_instruction_idx = target // 2
        else:
            self.pop()

    def JUMP_IF_FALSE_OR_POP(self, target):
        """If TOS is false, sets the bytecode counter to target and leaves TOS on the stack.
        Otherwise(TOS is true), TOS is popped."""
        if not self.top():
            self.frame.cur_instruction_idx = target // 2
        else:
            self.pop()

    def JUMP_ABSOLUTE(self, target):
        """Set bytecode counter to target."""
        self.frame.cur_instruction_idx = target // 2

    def FOR_ITER(self, delta):
        """
        TOS is an iterator. Call its __next__() method.
        If this yields a new value, push it on the stack(leaving the iterator below it).
        If the iterator indicates it is exhausted TOS is popped, and the byte code counter is incremented by delta.
        """
        TOS = self.top()
        try:
            self.log("TRY TO GET NEXT FROM GENERATOR")
            new_value = TOS.__next__()
            self.log("new_value", new_value)
            self.push(new_value)
            if len(self.frames) == 1:
                self.logging = False
        except StopIteration:
            self.log("ITERATION STOPED")
            self.pop()
            self.frame.cur_instruction_idx += delta // 2

    def LOAD_GLOBAL(self, namei):
        """Loads the global named co_names[namei] onto the stack."""
        self.push(self.find_associated(self.frame.co_names[namei], "globals"))

    def SETUP_LOOP(self, delta):
        """Pushes a block for a loop onto the block stack.
        The block spans from the current instruction with a size of delta bytes."""
        self.log(*self.frame.instructions)
        self.push_block('loop', delta)

    def POP_BLOCK(self):
        """Removes one block from the block stack.
        Per frame, there is a stack of blocks, denoting nested loops, try statements, and such.
        """
        return self.pop_block()

    def POP_EXCEPT(self):
        """Removes one block from the block stack. The popped block must be an exception handler block,
        as implicitly created when entering an except handler. In addition to popping extraneous values from the frame
        stack, the last three popped values are used to restore the exception state.
        """
        block = self.pop_block()
        tb, value, exctype = self.popn(3)

    def SETUP_EXCEPT(self, delta):
        """Pushes a try block from a try-except clause onto the block stack. delta points to the first except block."""
        self.push_block('except', delta // 2 + self.frame.cur_instruction_idx)

    def SETUP_FINALLY(self, delta):
        """Pushes a try block from a try-except clause onto the block stack. delta points to the finally block."""
        self.push_block('finally', delta // 2 + self.frame.cur_instruction_idx)

    def SET_ADD(self, i):
        """Calls set.add(TOS1[-i], TOS). Used to implement set comprehensions."""
        TOS = self.pop()
        self.frame.stack[-i].add(TOS)

    def LIST_APPEND(self, i):
        """Calls list.append(TOS[-i], TOS). Used to implement list comprehensions."""
        TOS = self.pop()
        self.frame.stack[-i].append(TOS)

    def MAP_ADD(self, i):
        """Calls dict.setitem(TOS1[-i], TOS, TOS1). Used to implement dict comprehensions."""
        TOS1, TOS = self.popn(2)
        self.frame.stack[-i][TOS] = TOS1

    """For all of the SET_ADD, LIST_APPEND and MAP_ADD instructions,
    while the added value or key / value pair is popped off, 
    the container object remains on the stack so that it is available for further iterations of the loop.
    """

def function_creator(name, code, globals_nm, vm, argc, kwargs_defaults):

    vm.log("defults", kwargs_defaults, "globals", globals_nm)
    vm.log("code.co_varnames", code.co_varnames)
    vm.log("code.co_name", code.co_name)
    #co_argcount is the number of positional arguments (including arguments with default values)
    vm.log("co_argcount", code.co_argcount)
    vm.log("code.co_kwonlyargcount", code.co_kwonlyargcount)
    vm.log("code.co_flags", code.co_flags)
    vm.log("code.co_freevars", code.co_freevars)
    vm.log("code.co_cellvars", code.co_cellvars)

    f_globals = dict(globals_nm)
    f_globals.update(vm.frame.locals)
    var_names = code.co_varnames
    f_locals = {"__name__": code.co_name}
    f_locals.update(kwargs_defaults.get("func_closure", {}))
    f_locals.update(kwargs_defaults.get("func_kwdefaults", {}))
    defaults = kwargs_defaults.get("func_defaults", ())
    args_used = False
    if code.co_flags & 0x04:
        #function uses the *arguments
        args_used = True

    kwargs_used = False
    if code.co_flags & 0x08:
        #function uses the **keywords
        kwargs_used = True

    def func(*args, **kwargs):
        if not args_used:
            check_if_ok(args, code, defaults)
            f_locals.update({var_names[i]: args[i] for i in range(len(args))})
            defaults_took = len(args) + len(defaults) - code.co_argcount
            vm.log(f_locals)
            vm.log(defaults_took)
            if defaults_took < 0:
                raise NotImplementedError
            for i in range(len(args), code.co_argcount):
                vm.log(defaults_took + i, defaults)
                vm.log(var_names[i])
                f_locals[var_names[i]] = defaults[defaults_took + i - len(args)]

            vm.log("ok")
            for key in var_names[:code.co_argcount]:
                if key in kwargs:
                    f_locals[key] = kwargs[key]
                    del kwargs[key]

            vm.log("okok")
            if kwargs_used:
                f_locals[var_names[code.co_argcount]] = kwargs
        else:
            f_locals.update({var_names[i]: args[i] for i in range(code.co_argcount)})
            args_left = args[code.co_argcount:]

            if kwargs_used:
                for key in var_names[code.co_argcount:code.co_argcount + code.co_kwonlyargcount]:
                    if key in kwargs:
                        f_locals[key] = kwargs[key]
                        del kwargs[key]
                f_locals[var_names[code.co_argcount + code.co_kwonlyargcount + 1]] = kwargs
            f_locals[var_names[code.co_argcount + code.co_kwonlyargcount]] = args_left

        vm.log(f_locals)
        frame = Frame(code, f_locals, f_globals, vm.frame)

        vm.log("new instructions", *list(dis.get_instructions(code)))
        return vm.run_frame(frame)

    func.__name__ = code.co_name
    func.__doc__ = code.co_consts[0] if code.co_consts else None
    func.__dict__ = {}
    vm.log(func)
    return func


class Function(object):

    def __init__(self, name, code, globals_nm, vm, argc, kwargs_defaults):
        self.__name__ = name
        vm.log("name", name, in_line=True)
        self.__doc__ = code.co_consts[0] if code.co_consts else None

        self.code = code
        self.f_globals = dict(globals_nm)
        self.f_globals.update(vm.frame.locals)
        self.f_globals.update({name: self})
        self.var_names = self.code.co_varnames
        self.f_locals = {"__name__": self.code.co_name}
        self.f_locals.update(kwargs_defaults.get("func_closure", {}))
        self.f_locals.update(kwargs_defaults.get("func_kwdefaults", {}))
        self.defaults = kwargs_defaults.get("func_defaults", ())
        self.vm = vm

        self.args_used = False
        if self.code.co_flags & 0x04:
            # function uses the *arguments
            self.args_used = True

        self.kwargs_used = False
        if self.code.co_flags & 0x08:
            # function uses the **keywords
            self.kwargs_used = True
        self.is_generator = False
        if self.code.co_flags & 0x20:
            # the function is a generator
            self.is_generator = True

    def __call__(self, *args, **kwargs):
        self.vm.log("Function called")
        self.handle_args_kwargs(*args, **kwargs)
        self.vm.log("arguments parsed")
        self.frame = Frame(self.code, self.f_locals, self.f_globals, self.vm.frame)
        self.vm.log("frame created")
        self.vm.log("new instructions!!!!!!!!!!!", *list(dis.get_instructions(self.code)))

        if self.is_generator:
            self.vm.log("return generator")
            return Generator(self.frame, self.vm)
        else:
            self.vm.log("function is not generator so call it ")
            return self.vm.run_frame(self.frame)

    def handle_args_kwargs(self, *args, **kwargs):
        self.vm.log("try to handle args and kwargs")
        if not self.args_used:
            check_if_ok(args, self.code, self.defaults)
            self.f_locals.update({self.var_names[i]: args[i] for i in range(len(args))})
            defaults_took = len(args) + len(self.defaults) - self.code.co_argcount
            self.vm.log(self.f_locals)
            self.vm.log(defaults_took)
            if defaults_took < 0:
                raise NotImplementedError
            for i in range(len(args), self.code.co_argcount):
                self.vm.log(defaults_took + i, self.defaults)
                self.vm.log(self.var_names[i])
                self.f_locals[self.var_names[i]] = self.defaults[defaults_took + i - len(args)]

            for key in self.var_names[:self.code.co_argcount]:
                if key in kwargs:
                    self.f_locals[key] = kwargs[key]
                    del kwargs[key]

            self.vm.log("okok")
            if self.kwargs_used:
                self.f_locals[self.var_names[self.code.co_argcount]] = kwargs
        else:
            self.f_locals.update({self.var_names[i]: args[i] for i in range(self.code.co_argcount)})
            args_left = args[self.code.co_argcount:]

            if self.kwargs_used:
                for key in self.var_names[self.code.co_argcount:self.code.co_argcount + self.code.co_kwonlyargcount]:
                    if key in kwargs:
                        self.f_locals[key] = kwargs[key]
                        del kwargs[key]
                self.f_locals[self.var_names[self.code.co_argcount + self.code.co_kwonlyargcount + 1]] = kwargs
            self.f_locals[self.var_names[self.code.co_argcount + self.code.co_kwonlyargcount]] = args_left


class Generator(object):

    def __init__(self, frame, vm):
        self.frame = frame
        self.vm = vm
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        finished = self.vm.run_frame(self.frame)
        if finished:
            raise StopIteration
        self.vm.log("yielded value", self.vm.value)
        return self.vm.value


def check_if_ok(args, code, defults):
    if len(args) > code.co_argcount:
        raise TypeError("{:s}() takes from {:d} to {:d} positional arguments but {:d}\
 were given".format(code.co_name,
                             code.co_argcount - len(defults),
                             code.co_argcount,
                             len(args)))
    if len(args) + len(defults) < code.co_argcount:
        missing = code.co_argcount - len(defults) - len(args)
        reason = "{:s}() missing {:d}\
         required positional argument".format(code.co_name, missing)
        if missing == 1:
            reason += ": '{:s}'".format(code.co_varnames[len(args)])
        else:
            reason += "s: '{:s}' and '{:s}'".format("', '".join(code.co_varnames[len(args):len(args) + missing - 1]
                                                                ), code.co_varnames[len(args) + missing - 1])
        raise TypeError(reason)


if __name__== "__main__":
    codeddd = compile(r"""
a = [[1,2, 3, 4], [5, 6,7 ,8 ], [9 ,10]]
z = tuple(x for x in a)
print(z)
""", '<stdin>', 'exec')

    VirtualMachine(logging=True).run(codeddd)
