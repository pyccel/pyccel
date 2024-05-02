    def view_namespace(namespace, entry):
        """
        Print contents of a namespace.

        Parameters
        ----------
        namespace: dict
            Dictionary that represents the current namespace, usually attached to a BasicParser object.

        entry: str
            Key of interest.

        """

        # TODO improve

        try:
            from tabulate import tabulate

            table = []
            for (k, v) in namespace[entry].items():
                k_str = '{}'.format(k)
                if entry == 'imports':
                    if v is None:
                        v_str = '*'
                    else:
                        v_str = '{}'.format(v)
                elif entry == 'variables':
                    v_str = '{}'.format(type(v))
                else:
                    raise NotImplementedError('TODO')

                line = [k_str, v_str]
                table.append(line)
            headers = ['module', 'target']

#            txt = tabulate(table, headers, tablefmt="rst")

            txt = tabulate(table, tablefmt='rst')
            print (txt)
        except NotImplementedError:

            print ('------- namespace.{} -------'.format(entry))
            for (k, v) in namespace[entry].items():
                print ('{var} \t :: \t {value}'.format(var=k, value=v))
            print ('-------------------------')
