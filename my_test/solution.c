

// if step < 0:
//     loop_var = self.scope.get_temporary_variable(NativeInteger(), name='i')
//     loop_start = FunctionCall('len', [array]) - 1
//     loop_condition = BinaryOp(loop_var, '>=', 0)
//     loop_update = Assignment(loop_var, loop_var + step)
// else:
//     loop_var = self.scope.get_temporary_variable(NativeInteger(), name='i')
//     loop_start = LiteralInteger(0)
//     loop_condition = BinaryOp(loop_var, '<', FunctionCall('len', [array]))
//     loop_update = Assignment(loop_var, loop_var + step)

// real_index = Conditional((loop_var < 0), FunctionCall('len', [array]) + loop_var, loop_var)


printf("%s", "[ ");
    
    for (index = (step < 0) ? a.shape[0] - 1 : 0; 
         (step < 0) ? (index >= 0) : (index < a.shape[0]); 
         index += step) 
    {
        int64_t real_index = (step < 0) ? ((index < 0) ? a.shape[0] + index : index) : index;
        printf("%.12lf ", GET_ELEMENT(a, nd_double, real_index));
    }
    
    printf("%s", "]\n");
