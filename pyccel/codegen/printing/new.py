Plan:



c_binding():

generate_convert_array():

generate_convert_scalar():

print_Interface():

Print_FunctionDef():
    --> generate new wrapper name
    --> collect used_names #it can be droped
    --> collect wrapper variable
    --> collect key_words
    --> loop on all function arguments :
        --> generate convert function
        --> generate new_args (needed c binding) #it can be droped

    --> generate PyArg_ParseNode
    --> generate static functioncall

    --> loop on all  function results :
        --> generate convert function or use the old principe

    --> build the FunctionDef
    --> print the FunctionDef

Print_Module():

# --------------------------------------------------------------------
#                       Helper functions
# --------------------------------------------------------------------

def get_new_name(self, used_names, requested_name):
    """
    """
    if requested_name not in used_names:
        used_names.add(requested_name)
        return requested_name
    else:
        incremented_name, _ = create_incremented_string(used_names, prefix=requested_name)
        return incremented_name

def get_wrapper_name(self, used_names, function):
    """
    """
    name = function.name
    wrapper_name = self.get_new_name(used_names.union(self._global_names), name+"_wrapper")

    self._function_wrapper_names[func.name] = wrapper_name
    self._global_names.add(wrapper_name)
    used_names.add(wrapper_name)

    return wrapper_name

def get_new_PyObject(self, name, used_names):
    """
    """
    return Variable(dtype      = PyccelPyObject(),
                    name       = self.get_new_name(used_names, name),
                    is_pointer = True)


def get_wrapper_arguments(self, used_names)
    """
    """
    python_func_args    = self.get_new_PyObject("args"  , used_names)
    python_func_kwargs  = self.get_new_PyObject("kwargs", used_names)
    python_func_selfarg = self.get_new_PyObject("self"  , used_names)

    return [python_func_selfarg, python_func_args, python_func_kwargs]


def _print_Interface(self, expr):
    # TODO

def _print_FunctionDef(self, expr):
    # Save all used names
    used_names = set([a.name for a in expr.arguments]
                   + [r.name for r in expr.results]
                   + [expr.name])

    # Find a name for the wrapper function
    wrapper_name = self.get_wrapper_name(used_names, expr)

    # Collect arguments and results
    wrapper_args    = get_wrapper_arguments(used_names)
    wrapper_results = [self.get_new_PyObject("result", used_names)]

    arg_names         = [a.name for a in expr.arguments]
    keyword_list_name = self.get_new_name(used_names,'kwlist')
    keyword_list      = PyArgKeywords(keyword_list_name, arg_names)

    func_args = []
    wrapper_body = [keyword_list]
    convert_function_dict = {}

    for arg in expr.arguments:
        convert_function = None #TODO Generate convert function
        func_args.append(None) #TODO Bind_C_Arg
        convert_function_dict[arg] = convert_function.name

    parse_node = PyArg_ParseTupleNode() #TODO

    wrapper_body.append(If(IfSection(PyccelNot(parse_node),
                        [Return([Nil()])])))

    static_function = None #TODO Generate Bind_C_Arg functionCall

    function_call = FunctionCall(static_function, func_args)
    
    if len(expr.results) > 0:
        results       = expr.results if len(expr.results)>1 else expr.results[0]
        function_call = Assign(results, function_call)
    
    wrapper_body.append(func_call)

    for res in expr.results:
        convert_function = None: #TODO
        convert_function_dict[arg] = convert_function.name

    build_node = PyBuildValueNode(convert_function_dict, expr.results)

    wrapper_body.append(AliasAssign(wrapper_results[0], build_node))

    wrapper_function = FunctionDef(name        = wrapper_name,
                                   arguments   = wrapper_args,
                                   results     = wrapper_results,
                                   body        = wrapper_body,
                                   local_varts = tuple(func_args + expr.results))
    
    return CCodePrinter._print_FunctionDef(self, wrapper_func)

def _print_Module(self, expr):
    self._global_names = set(f.name for f in expr.funcs)
    self._module_name  = expr.name
    
    .
    .
    .
    .